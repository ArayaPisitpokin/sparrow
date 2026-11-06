// Contact-aware LBF construction ported from the 2026-08-27 playground's vendored
// sparrow (tm3:~/xerow-playground-20260827/vendor/sparrow). All features are gated by
// XEROW_LBF_* environment variables; with none set this file behaves exactly like
// upstream 881cdcb. Best strict scratch result of the contact constructor family:
// 6099.605 mm in a 1200 s cold solve (gap doc, Evidence Log).
use crate::eval::lbf_evaluator::LBFEvaluator;
use crate::eval::sample_eval::{SampleEval, SampleEvaluator};
use crate::sample::coord_descent::{refine_coord_desc, CDConfig};
use crate::sample::search::{search_placement, SampleConfig};
use crate::util::assertions;
use itertools::Itertools;
use jagua_rs::entities::Instance;
use jagua_rs::geometry::geo_enums::RotationRange;
use jagua_rs::geometry::geo_traits::TransformableFrom;
use jagua_rs::geometry::primitives::Point;
use jagua_rs::geometry::{DTransformation, Transformation};
use jagua_rs::probs::spp::entities::{SPInstance, SPPlacement, SPProblem};
use jagua_rs::probs::spp::entities::SPSolution;
use jagua_rs::Instant;
use log::debug;
use ordered_float::OrderedFloat;
use std::cmp::Reverse;
use std::iter;
use rand::prelude::IndexedRandom;
use rand::rngs::Xoshiro256PlusPlus;

pub struct LBFBuilder {
    pub instance: SPInstance,
    pub prob: SPProblem,
    pub rng: Xoshiro256PlusPlus,
    pub sample_config: SampleConfig,
}

impl LBFBuilder {
    pub fn new(
        instance: SPInstance,
        rng: Xoshiro256PlusPlus,
        sample_config: SampleConfig,
    ) -> Self {
        let prob = SPProblem::new(instance.clone());

        Self {
            instance,
            prob,
            rng,
            sample_config,
        }
    }

    /// Start from a feasible partial layout and place every still-missing item
    /// with the ordinary largest-first LBF constructor.
    pub fn from_partial_solution(
        instance: SPInstance,
        rng: Xoshiro256PlusPlus,
        sample_config: SampleConfig,
        solution: &SPSolution,
    ) -> Self {
        let mut builder = Self::new(instance, rng, sample_config);
        builder.prob.restore(solution);
        builder
    }

    pub fn construct(mut self) -> Self {
        let start = Instant::now();
        let n_items = self.instance.items.len();
        let sorted_item_types = (0..n_items)
            .sorted_by_cached_key(|id| {
                let item_shape = self.instance.item(*id).shape_cd.as_ref();
                let convex_hull_area = item_shape.surrogate().convex_hull_area;
                let diameter = item_shape.diameter;
                Reverse(OrderedFloat(convex_hull_area * diameter))
            })
            .collect_vec();

        // Upstream places every demanded copy of a type consecutively. For
        // garment markers that can separate complementary left/right pieces:
        // all copies of one panel are committed before its almost equally
        // sized mate is considered. The opt-in bucket mode preserves the
        // largest-first ordering between size bands, but round-robins copies
        // within a band of similarly scored item types.
        let order_bucket = std::env::var("XEROW_LBF_ORDER_BUCKET")
            .ok()
            .and_then(|value| value.parse::<f32>().ok())
            .filter(|value| value.is_finite() && *value > 1.0);
        let mut sorted_item_indices = Vec::with_capacity(
            self.prob.item_demand_qtys.iter().sum(),
        );
        if let Some(bucket_ratio) = order_bucket {
            let score = |id: usize| {
                let shape = self.instance.item(id).shape_cd.as_ref();
                shape.surrogate().convex_hull_area * shape.diameter
            };
            let mut start = 0;
            while start < sorted_item_types.len() {
                let ceiling = score(sorted_item_types[start]);
                let mut end = start + 1;
                while end < sorted_item_types.len()
                    && ceiling / score(sorted_item_types[end]).max(f32::EPSILON)
                        <= bucket_ratio
                {
                    end += 1;
                }
                let band = &sorted_item_types[start..end];
                let max_qty = band
                    .iter()
                    .map(|id| self.prob.item_demand_qtys[*id])
                    .max()
                    .unwrap_or(0);
                for copy_index in 0..max_qty {
                    sorted_item_indices.extend(
                        band.iter()
                            .copied()
                            .filter(|id| self.prob.item_demand_qtys[*id] > copy_index),
                    );
                }
                start = end;
            }
        } else {
            sorted_item_indices.extend(sorted_item_types.iter().copied().flat_map(|id| {
                let missing_qty = self.prob.item_demand_qtys[id];
                iter::repeat_n(id, missing_qty)
            }));
        }

        debug!("[CONSTR] placing items in order: {:?}",sorted_item_indices);

        let lookahead = std::env::var("XEROW_LBF_LOOKAHEAD")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .filter(|value| *value > 1)
            .unwrap_or(1);
        if lookahead == 1 {
            for item_id in sorted_item_indices {
                self.place_item(item_id);
            }
        } else {
            let mut remaining = self.prob.item_demand_qtys.clone();
            let mut copies_left: usize = remaining.iter().sum();
            while copies_left > 0 {
                let candidate_ids = sorted_item_types
                    .iter()
                    .copied()
                    .filter(|id| remaining[*id] > 0)
                    .take(lookahead)
                    .collect_vec();
                let current_right = self
                    .prob
                    .layout
                    .placed_items
                    .values()
                    .map(|placed| placed.shape.bbox.x_max)
                    .max_by(f32::total_cmp)
                    .unwrap_or(self.prob.layout.container.outer_cd.bbox.x_min);
                let fabric_height = self.prob.layout.container.outer_cd.bbox.height();
                let mut candidates = Vec::with_capacity(candidate_ids.len());
                for item_id in candidate_ids {
                    if let Some(placement) = self.find_placement(item_id) {
                        let item = self.instance.item(item_id);
                        let mut placed_shape = item.shape_cd.as_ref().clone();
                        placed_shape.transform_from(
                            item.shape_cd.as_ref(),
                            &Transformation::from(placement.d_transf),
                        );
                        let projected_right = current_right.max(placed_shape.bbox.x_max);
                        // The lower bound on the strip-length contribution of a
                        // piece is area/fabric_height. Prefer candidates whose
                        // actual envelope growth is closest to that ideal.
                        let incremental_waste = projected_right - current_right
                            - item.shape_cd.area / fabric_height;
                        candidates.push((OrderedFloat(incremental_waste), item_id, placement));
                    }
                }
                if let Some((_waste, item_id, placement)) =
                    candidates.into_iter().min_by_key(|candidate| candidate.0)
                {
                    self.prob.place_item(placement);
                    remaining[item_id] -= 1;
                    copies_left -= 1;
                } else {
                    self.prob.change_strip_width(self.prob.strip_width() * 1.2);
                    assert!(
                        assertions::strip_width_is_in_check(&self.prob),
                        "strip-width is running away during look-ahead construction"
                    );
                }
            }
        }

        self.prob.fit_strip();
        if std::env::var_os("XEROW_LBF_DIAGNOSTICS").is_some() {
            eprintln!(
                "[XEROW_LBF] width={:.6} placed={} elapsed_s={:.6}",
                self.prob.strip_width(),
                self.prob.layout.placed_items.len(),
                start.elapsed().as_secs_f32(),
            );
        }
        debug!("[CONSTR] placed all items in width: {:.3} (in {:?})",self.prob.strip_width(), start.elapsed());
        self
    }

    fn place_item(&mut self, item_id: usize) {
        match self.find_placement(item_id) {
            Some(p_opt) => {
                self.prob.place_item(p_opt);
                debug!("[CONSTR] placing item {}/{} with id {} at [{}]",self.prob.layout.placed_items.len(),self.instance.total_item_qty(),p_opt.item_id,p_opt.d_transf);
            }
            None => {
                debug!("[CONSTR] failed to place item with id {}, expanding strip width",item_id);
                self.prob.change_strip_width(self.prob.strip_width() * 1.2);
                assert!(assertions::strip_width_is_in_check(&self.prob), "strip-width is running away (>{:.3}), item {item_id} does not seem to fit into the strip", self.prob.strip_width());          
                self.place_item(item_id);
            }
        }
    }

    fn find_placement(&mut self, item_id: usize) -> Option<SPPlacement> {
        let layout = &self.prob.layout;
        let item = self.instance.item(item_id);
        let contact_refine = std::env::var_os("XEROW_LBF_CONTACT_REFINE").is_some();
        let contact_refine_top = std::env::var("XEROW_LBF_CONTACT_REFINE_TOP")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .filter(|value| *value > 0)
            .unwrap_or(1);
        let contact_vertex_samples = std::env::var("XEROW_LBF_VERTEX_CONTACT_SAMPLES")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(0);
        let salient_vertex_limit = std::env::var("XEROW_LBF_SALIENT_VERTICES")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .filter(|value| *value >= 4)
            .unwrap_or(32);
        let contact_candidates = std::env::var_os("XEROW_LBF_CONTACT_SAMPLES")
            .map(|_| {
                let epsilon = std::env::var("XEROW_LBF_CONTACT_EPSILON")
                    .ok()
                    .and_then(|value| value.parse::<f32>().ok())
                    .filter(|value| value.is_finite() && *value > 0.0)
                    .unwrap_or(0.05);
                let rotations: Vec<f32> = match &item.allowed_rotation {
                    RotationRange::None => vec![0.0],
                    RotationRange::Discrete(values) => values.clone(),
                    RotationRange::Continuous => (0..16)
                        .map(|index| index as f32 * std::f32::consts::TAU / 16.0)
                        .collect(),
                };
                let container = layout.container.outer_cd.bbox;
                let mut evaluator = LBFEvaluator::new(layout, item);
                let mut shape_buffer = item.shape_cd.as_ref().clone();
                let fixed_contacts = layout
                    .placed_items
                    .values()
                    .map(|placed| {
                        (
                            placed.shape.centroid(),
                            salient_vertices(&placed.shape.vertices, salient_vertex_limit),
                        )
                    })
                    .filter(|(_, vertices)| !vertices.is_empty())
                    .collect_vec();
                let retain_count = if contact_refine {
                    contact_refine_top
                } else {
                    1
                };
                let mut best_candidates: Vec<(DTransformation, SampleEval)> = Vec::new();
                for rotation in rotations {
                    let rotated_bbox = shape_buffer
                        .transform_from(
                            item.shape_cd.as_ref(),
                            &Transformation::from_rotation(rotation),
                        )
                        .bbox;
                    let moving_contacts =
                        salient_vertices(&shape_buffer.vertices, salient_vertex_limit);
                    for _ in 0..contact_vertex_samples {
                        let (Some((fixed_center, fixed_vertices)), Some(source)) = (
                            fixed_contacts.choose(&mut self.rng),
                            moving_contacts.choose(&mut self.rng),
                        ) else {
                            break;
                        };
                        let Some(target) = fixed_vertices.choose(&mut self.rng) else {
                            continue;
                        };
                        let mut outward = (target.0 - fixed_center.0, target.1 - fixed_center.1);
                        let norm = outward.0.hypot(outward.1);
                        if norm > f32::EPSILON {
                            outward.0 /= norm;
                            outward.1 /= norm;
                        } else {
                            outward = (1.0, 0.0);
                        }
                        let transformation = DTransformation::new(
                            rotation,
                            (
                                target.0 - source.0 + epsilon * outward.0,
                                target.1 - source.1 + epsilon * outward.1,
                            ),
                        );
                        let evaluation = evaluator.evaluate_sample(
                            transformation,
                            best_candidates.first().map(|(_, evaluation)| *evaluation),
                        );
                        if matches!(evaluation, SampleEval::Clear { .. }) {
                            best_candidates.push((transformation, evaluation));
                            best_candidates.sort_unstable_by(|a, b| a.1.cmp(&b.1));
                            best_candidates.truncate(retain_count);
                        }
                    }
                    let mut x_axes = vec![
                        container.x_min + epsilon - rotated_bbox.x_min,
                        container.x_max - epsilon - rotated_bbox.x_max,
                    ];
                    let mut y_axes = vec![
                        container.y_min + epsilon - rotated_bbox.y_min,
                        container.y_max - epsilon - rotated_bbox.y_max,
                    ];
                    for (_, placed) in layout.placed_items.iter() {
                        let bbox = placed.shape.bbox;
                        x_axes.extend([
                            bbox.x_max + epsilon - rotated_bbox.x_min,
                            bbox.x_min - epsilon - rotated_bbox.x_max,
                        ]);
                        y_axes.extend([
                            bbox.y_max + epsilon - rotated_bbox.y_min,
                            bbox.y_min - epsilon - rotated_bbox.y_max,
                        ]);
                    }
                    x_axes.sort_unstable_by(f32::total_cmp);
                    x_axes.dedup_by(|a, b| (*a - *b).abs() < 1e-4);
                    y_axes.sort_unstable_by(f32::total_cmp);
                    y_axes.dedup_by(|a, b| (*a - *b).abs() < 1e-4);
                    for &x in &x_axes {
                        for &y in &y_axes {
                            let transformation = DTransformation::new(rotation, (x, y));
                            let evaluation = evaluator.evaluate_sample(
                                transformation,
                                best_candidates.first().map(|(_, evaluation)| *evaluation),
                            );
                            if matches!(evaluation, SampleEval::Clear { .. }) {
                                best_candidates.push((transformation, evaluation));
                                best_candidates.sort_unstable_by(|a, b| a.1.cmp(&b.1));
                                best_candidates.truncate(retain_count);
                            }
                        }
                    }
                }
                best_candidates
            })
            .unwrap_or_default();

        // Bounding-box contact axes cheaply expose useful piece-to-piece
        // neighborhoods, but a concave garment outline can often slide much
        // farther once it is in that neighborhood.  Refine the best contact
        // candidate against the exact collision detector when explicitly
        // requested.  This is the constructor analogue of a shoot-and-slide
        // or "exact fit" pass; discrete rotations remain locked.
        let contact_best = contact_candidates
            .into_iter()
            .map(|candidate| {
                if !contact_refine {
                    return candidate;
                }
                let min_dim =
                    f32::min(item.shape_cd.bbox.width(), item.shape_cd.bbox.height());
                let mut evaluator = LBFEvaluator::new(layout, item);
                let coarse = refine_coord_desc(
                    candidate,
                    &mut evaluator,
                    CDConfig {
                        t_step_init: min_dim * 0.25,
                        t_step_limit: min_dim * 0.01,
                        r_step_init: 0.0,
                        r_step_limit: 0.0,
                        wiggle: false,
                    },
                    &mut self.rng,
                );
                refine_coord_desc(
                    coarse,
                    &mut evaluator,
                    CDConfig {
                        t_step_init: min_dim * 0.01,
                        t_step_limit: min_dim * 0.0005,
                        r_step_init: 0.0,
                        r_step_limit: 0.0,
                        wiggle: false,
                    },
                    &mut self.rng,
                )
            })
            .min_by(|a, b| a.1.cmp(&b.1));

        let evaluator = LBFEvaluator::new(layout, item);

        let (best_sample, _) = search_placement(layout, item, None, evaluator, self.sample_config, &mut self.rng, None);

        let best_sample = match (contact_best, best_sample) {
            (Some(contact), Some(random)) => Some(if contact.1 < random.1 { contact } else { random }),
            (Some(contact), None) => Some(contact),
            (None, random) => random,
        };

        match best_sample {
            Some((d_transf, SampleEval::Clear { .. })) => {
                Some(SPPlacement { item_id, d_transf })
            }
            _ => None
        }
    }
}

/// Select geometrically informative contour vertices without inheriting the
/// arbitrary tessellation density of the source CAD file. Extrema are always
/// retained; the remaining vertices are ranked by deviation from the chord
/// joining their immediate neighbors.
fn salient_vertices(vertices: &[Point], limit: usize) -> Vec<Point> {
    if vertices.len() <= limit {
        return vertices.to_vec();
    }

    let x_min = vertices.iter().map(|point| point.0).min_by(f32::total_cmp).unwrap();
    let x_max = vertices.iter().map(|point| point.0).max_by(f32::total_cmp).unwrap();
    let y_min = vertices.iter().map(|point| point.1).min_by(f32::total_cmp).unwrap();
    let y_max = vertices.iter().map(|point| point.1).max_by(f32::total_cmp).unwrap();
    let mut scored = vertices
        .iter()
        .enumerate()
        .map(|(index, point)| {
            let previous = vertices[(index + vertices.len() - 1) % vertices.len()];
            let next = vertices[(index + 1) % vertices.len()];
            let chord = (next.0 - previous.0, next.1 - previous.1);
            let chord_length = chord.0.hypot(chord.1);
            let deviation = if chord_length > f32::EPSILON {
                ((point.0 - previous.0) * chord.1
                    - (point.1 - previous.1) * chord.0)
                    .abs()
                    / chord_length
            } else {
                0.0
            };
            let is_extreme =
                point.0 == x_min || point.0 == x_max || point.1 == y_min || point.1 == y_max;
            (if is_extreme { f32::INFINITY } else { deviation }, *point)
        })
        .collect_vec();
    scored.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
    scored.truncate(limit);
    scored.into_iter().map(|(_, point)| point).collect()
}

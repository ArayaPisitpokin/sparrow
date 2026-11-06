use crate::eval::sep_evaluator::SeparationEvaluator;
use crate::quantify::tracker::CollisionTracker;
use crate::sample::search;
use crate::sample::search::SampleConfig;
use crate::util::assertions::tracker_matches_layout;
use crate::FMT;
use itertools::Itertools;
use jagua_rs::entities::{Instance, PItemKey};
use jagua_rs::geometry::DTransformation;
use jagua_rs::probs::spp::entities::{SPInstance, SPPlacement, SPProblem, SPSolution};
use log::debug;
use rand::prelude::SliceRandom;
use rand::RngExt;
use crate::optimizer::anneal::AnnealMap;
use crate::eval::sample_eval::SampleEval;
use crate::small::SmallPieceConfig;
use std::iter::Sum;
use std::ops::AddAssign;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use rand::rngs::Xoshiro256PlusPlus;
use tap::Tap;

pub struct SeparatorWorker {
    pub instance: SPInstance,
    pub prob: SPProblem,
    pub ct: CollisionTracker,
    pub rng: Xoshiro256PlusPlus,
    pub sample_config: SampleConfig,
    /// Grouped-orientation mode: per-item rotation locks (item_id -> locked).
    pub locked_items: Option<Arc<[bool]>>,
    /// Positional freeze registry (see `Separator::frozen`); never move these.
    pub frozen: Option<Arc<std::collections::HashSet<(usize, i64, i64, i64)>>>,
    /// Whether the locks currently apply (false during an anneal window).
    pub locks_active: Arc<AtomicBool>,
    /// Anneal consistency-pressure map for this pass (None outside anneal).
    pub anneal: Option<Arc<AnnealMap>>,
    /// Small-piece mechanisms A (clear-sample bias) and M (voluntary moves); see `crate::small`.
    pub small: SmallPieceConfig,
    /// item_id -> is "small" (below the large-item cutoff). Always populated.
    pub small_items: Arc<[bool]>,
    /// Small-piece mechanism L: lane centres for large items (empty = off).
    pub lane_centers: Arc<[f32]>,
}

impl SeparatorWorker {
    pub fn load(&mut self, sol: &SPSolution, ct: &CollisionTracker, anneal: Option<Arc<AnnealMap>>) {
        // restores the state of the worker to the given solution and accompanying tracker
        debug_assert!(sol.strip_width() == self.prob.strip_width());
        self.prob.restore(sol);
        self.ct = ct.clone();
        self.anneal = anneal;
    }

    /// Algorithm 5 from https://doi.org/10.48550/arXiv.2509.13329
    ///
    /// Anneal mode adds the consistency-disagreeing copies to the candidate set
    /// (pressure must bind on collision-free pieces too) and biases every
    /// candidate pose by the per-class anneal penalty.
    pub fn move_items(&mut self) -> SepStats {
        // Collect all colliding items (plus anneal move-candidates and, under small-piece
        // mechanism M, a random draw of collision-free small items that volunteer for a
        // positional move) in a random order.
        let voluntary_active = self.small.voluntary_moves_active();
        let mut candidates: Vec<(PItemKey, bool)> = Vec::with_capacity(self.prob.layout.placed_items.len());
        for (pk, pi) in self.prob.layout.placed_items.iter() {
            if crate::optimizer::separator::pi_frozen(&self.frozen, pi) {
                continue;   // frozen copies never move; colliders must move around them
            }
            let forced = self.ct.get_loss(pk) > 0.0
                || self.anneal.as_ref().and_then(|m| m.get(pk)).is_some_and(|e| e.move_candidate);
            if forced {
                candidates.push((pk, false));
            } else if voluntary_active
                && self.small_items[pi.item_id]
                && self.rng.random::<f32>() < self.small.clear_move_prob
            {
                candidates.push((pk, true));
            }
        }
        candidates.shuffle(&mut self.rng);

        let mut total_moves = 0;
        let mut total_evals = 0;
        let mut voluntary_moves = 0;

        // Give each colliding item the opportunity to move to a better (eval) position
        for &(pk, voluntary) in candidates.iter() {
            let anneal_entry = self.anneal.as_ref().and_then(|m| m.get(pk)).copied();
            // First check if the item is still colliding (or under anneal pressure, or volunteered)
            if self.ct.get_loss(pk) > 0.0 || anneal_entry.is_some_and(|e| e.move_candidate) || voluntary {
                let item_id = self.prob.layout.placed_items[pk].item_id;
                let item = self.instance.item(item_id);
                let current_dt = self.prob.layout.placed_items[pk].d_transf;

                // Grouped-orientation mode: locked items keep their current rotation
                // once locks are active (inactive during an anneal window). One
                // Option check + one slice index + one relaxed atomic load per move.
                let rot_lock = match self.locked_items.as_deref() {
                    Some(locked)
                        if locked[item_id] && self.locks_active.load(Ordering::Relaxed) =>
                    {
                        Some(current_dt.rotation())
                    }
                    _ => None,
                };

                // Create an 'evaluator' to perform collision detection and collision quantification of the samples during the search.
                // Small-piece mechanism A: the clear samples of small items are ranked by position.
                let is_small = self.small_items[item_id];
                let clear_bias = if is_small { self.small.clear_bias } else { 0.0 };
                let sample_config = if is_small && self.small.extra_container_samples > 0 {
                    SampleConfig {
                        n_container_samples: self.sample_config.n_container_samples + self.small.extra_container_samples,
                        ..self.sample_config
                    }
                } else {
                    self.sample_config
                };
                let evaluator = SeparationEvaluator::new(&self.prob.layout, item, pk, &self.ct)
                    .with_clear_bias(clear_bias);

                // Small-piece mechanism L: large items pay a lane-deviation penalty on
                // every sample (both clear and colliding), scaled to the item's own
                // collision magnitude — the panel-side analogue of the anneal pressure.
                let lane_penalty = if !is_small && self.small.lane_weight > 0.0 && !self.lane_centers.is_empty() {
                    let h = self.prob.layout.container.outer_cd.bbox.height();
                    let unit = self.small.lane_weight * item.shape_cd.surrogate().convex_hull_area.sqrt() / h.max(1.0);
                    let c = item.shape_cd.centroid();
                    Some((unit, (c.0, c.1)))
                } else {
                    None
                };

                // Perform the search for a better position for the item; anneal
                // pressure biases each candidate pose by its class penalty.
                let (best_sample, n_evals) = match (anneal_entry, lane_penalty) {
                    (Some(e), _) if e.penalty != [0.0; 2] => {
                        let wrapped = AnnealEval { inner: evaluator, penalty: e.penalty };
                        search::search_placement(&self.prob.layout, item, Some(pk), wrapped, sample_config, &mut self.rng, rot_lock)
                    }
                    (_, Some((unit, c0))) => {
                        let wrapped = LaneEval { inner: evaluator, unit, centroid0: c0, lanes: self.lane_centers.clone() };
                        search::search_placement(&self.prob.layout, item, Some(pk), wrapped, sample_config, &mut self.rng, rot_lock)
                    }
                    _ => search::search_placement(&self.prob.layout, item, Some(pk), evaluator, sample_config, &mut self.rng, rot_lock),
                };

                let (new_dt, _eval) = best_sample.expect("search_placement should always return a sample");
                total_evals += n_evals;

                // A volunteer that found nothing better than where it sits stays put.
                if voluntary
                    && new_dt.rotation() == current_dt.rotation()
                    && new_dt.translation() == current_dt.translation()
                {
                    continue;
                }

                // Move the item to the new position
                self.move_item(pk, new_dt);
                total_moves += 1;
                if voluntary {
                    voluntary_moves += 1;
                }
            }
        }
        SepStats { total_moves, total_evals, voluntary_moves }
    }

    pub fn move_item(&mut self, pk: PItemKey, d_transf: DTransformation) -> PItemKey {
        debug_assert!(tracker_matches_layout(&self.ct, &self.prob.layout));

        let item = self.instance.item(self.prob.layout.placed_items[pk].item_id);

        let (old_l, old_w_l) = (self.ct.get_loss(pk), self.ct.get_weighted_loss(pk));

        debug_assert!(old_l > 0.0 || self.anneal.is_some() || self.small.voluntary_moves_active(), "Item with key {:?} should be colliding, but has no loss: {}", pk, FMT().fmt2(old_l));
        debug_assert!(old_w_l > 0.0 || self.anneal.is_some() || self.small.voluntary_moves_active(), "Item with key {:?} should be colliding, but has no weighted loss: {}", pk, FMT().fmt2(old_w_l));

        // First removing the item and subsequently place it in its new position
        let old_placement = self.prob.remove_item(pk);
        let new_placement = SPPlacement { d_transf, item_id: item.id };
        let new_pk = self.prob.place_item(new_placement);

        // Update the collision tracker to reflect the changes
        self.ct.register_item_move(&self.prob.layout, pk, new_pk);

        let (new_l, new_w_l) = (self.ct.get_loss(new_pk), self.ct.get_weighted_loss(new_pk));

        debug!("Moved {:?} (l: {}, wl: {}) to {:?} (l+1: {}, wl+1: {})", old_placement, FMT().fmt2(old_l), FMT().fmt2(old_w_l), new_placement, FMT().fmt2(new_l), FMT().fmt2(new_w_l));
        debug_assert!(new_w_l <= old_w_l * 1.001 || self.anneal.is_some(), "weighted loss should never increase: {} > {}", FMT().fmt2(old_w_l), FMT().fmt2(new_w_l));
        debug_assert!(tracker_matches_layout(&self.ct, &self.prob.layout));

        new_pk
    }
}

/// Biases candidate-pose evaluations by a per-orientation-class penalty
/// (anneal consistency pressure). Bound pass-through is conservative: penalties
/// only add loss, so inner early-termination stays correct.
pub struct AnnealEval<E: crate::eval::sample_eval::SampleEvaluator> {
    pub inner: E,
    pub penalty: [f32; 2],
}

impl<E: crate::eval::sample_eval::SampleEvaluator> crate::eval::sample_eval::SampleEvaluator
    for AnnealEval<E>
{
    fn evaluate_sample(&mut self, dt: DTransformation, upper_bound: Option<SampleEval>) -> SampleEval {
        let p = self.penalty[crate::grouped::is_dir_180(dt.rotation()) as usize];
        match self.inner.evaluate_sample(dt, upper_bound) {
            SampleEval::Clear { loss } => SampleEval::Clear { loss: loss + p },
            SampleEval::Collision { loss } => SampleEval::Collision { loss: loss + p },
            SampleEval::Invalid => SampleEval::Invalid,
        }
    }
    fn n_evals(&self) -> usize {
        self.inner.n_evals()
    }
}

/// Small-piece mechanism L: adds `unit x |lane-deviation|` of the item's would-be
/// centroid to every sample evaluation of a large item. Penalties only add loss, so
/// inner early-termination stays conservative (like `AnnealEval`).
pub struct LaneEval<E: crate::eval::sample_eval::SampleEvaluator> {
    pub inner: E,
    pub unit: f32,
    pub centroid0: (f32, f32),
    pub lanes: Arc<[f32]>,
}

impl<E: crate::eval::sample_eval::SampleEvaluator> crate::eval::sample_eval::SampleEvaluator for LaneEval<E> {
    fn evaluate_sample(&mut self, dt: DTransformation, upper_bound: Option<SampleEval>) -> SampleEval {
        // centroid under dt: rotate the reference centroid, add the translation
        let (sin, cos) = dt.rotation().sin_cos();
        let cy = self.centroid0.0 * sin + self.centroid0.1 * cos + dt.translation().1;
        let dev = self
            .lanes
            .iter()
            .map(|l| (cy - l).abs())
            .fold(f32::INFINITY, f32::min);
        let p = self.unit * dev;
        match self.inner.evaluate_sample(dt, upper_bound) {
            SampleEval::Clear { loss } => SampleEval::Clear { loss: loss + p },
            SampleEval::Collision { loss } => SampleEval::Collision { loss: loss + p },
            SampleEval::Invalid => SampleEval::Invalid,
        }
    }
    fn n_evals(&self) -> usize {
        self.inner.n_evals()
    }
}

pub struct SepStats {
    pub total_moves: usize,
    pub total_evals: usize,
    /// Moves of collision-free small items (small-piece mechanism M); included in `total_moves`.
    pub voluntary_moves: usize,
}

impl Sum for SepStats {
    fn sum<I: Iterator<Item=SepStats>>(iter: I) -> Self {
        let mut total_moves = 0;
        let mut total_evals = 0;
        let mut voluntary_moves = 0;

        for report in iter {
            total_moves += report.total_moves;
            total_evals += report.total_evals;
            voluntary_moves += report.voluntary_moves;
        }

        SepStats { total_moves, total_evals, voluntary_moves }
    }
}

impl AddAssign for SepStats {
    fn add_assign(&mut self, other: Self) {
        self.total_moves += other.total_moves;
        self.total_evals += other.total_evals;
        self.voluntary_moves += other.voluntary_moves;
    }
}

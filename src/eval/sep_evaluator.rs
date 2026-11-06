use crate::eval::sample_eval::{SampleEval, SampleEvaluator};
use crate::eval::specialized_jaguars_pipeline::{collect_poly_collisions_in_detector_custom, SpecializedHazardCollector};
use crate::quantify::tracker::CollisionTracker;
use crate::small::positional_loss;
use jagua_rs::collision_detection::hazards::collector::HazardCollector;
use jagua_rs::entities::Item;
use jagua_rs::entities::Layout;
use jagua_rs::entities::PItemKey;
use jagua_rs::geometry::primitives::SPolygon;
use jagua_rs::geometry::DTransformation;

pub struct SeparationEvaluator<'a> {
    layout: &'a Layout,
    item: &'a Item,
    collector: SpecializedHazardCollector<'a>,
    shape_buff: SPolygon,
    n_evals: usize,
    /// Small-piece mechanism A: signed positional bias applied to collision-free
    /// samples (0 = upstream behaviour, all clear samples tie at loss 0).
    clear_bias: f32,
}

impl<'a> SeparationEvaluator<'a> {
    pub fn new(
        layout: &'a Layout,
        item: &'a Item,
        current_pk: PItemKey,
        ct: &'a CollisionTracker,
    ) -> Self {
        let collector = SpecializedHazardCollector::new(layout, ct, current_pk);

        Self {
            layout,
            item,
            collector,
            shape_buff: item.shape_cd.as_ref().clone(),
            n_evals: 0,
            clear_bias: 0.0,
        }
    }

    /// Ranks collision-free samples by `crate::small::positional_loss` with this bias.
    pub fn with_clear_bias(mut self, clear_bias: f32) -> Self {
        self.clear_bias = clear_bias;
        self
    }
}

impl<'a> SampleEvaluator for SeparationEvaluator<'a> {
    /// Evaluates a transformation. An upper bound can be provided to early terminate the process.
    /// Algorithm 7 from https://doi.org/10.48550/arXiv.2509.13329
    fn evaluate_sample(&mut self, dt: DTransformation, upper_bound: Option<SampleEval>) -> SampleEval {
        self.n_evals += 1;
        let cde = self.layout.cde();

        // Calculate an upper bound of quantification, above which samples are guaranteed to be rejected (because they are dominated by previous ones).
        let loss_bound = match upper_bound {
            Some(SampleEval::Collision { loss }) => loss,
            Some(SampleEval::Clear { .. }) => 0.0,
            _ => f32::INFINITY,
        };
        
        // Reload the hazard collector to prepare for a new query
        self.collector.reload(loss_bound);

        // Perform the collision detection and the collision quantification. The 'collector' will be populated with the results.
        collect_poly_collisions_in_detector_custom(cde, &dt, &mut self.shape_buff, self.item.shape_cd.as_ref(), &mut self.collector);
        
        if self.collector.early_terminate(&self.shape_buff) {
            // The total quantification of the collisions exceeded the upper bound and the process was terminated early.
            // Note that we might have exited before detecting/quantifying all collisions.
            // However, since we can asure that this sample will always be rejected, we don't need to spend any more time on it and just return `Invalid`.
            SampleEval::Invalid
        } else if self.collector.is_empty() {
            // No collisions detected, return clear. Small-piece mechanism A ranks the
            // clear samples of small items by position; `Clear` still always beats
            // `Collision`, so feasibility is never traded for position.
            let loss = if self.clear_bias != 0.0 {
                positional_loss(self.clear_bias, self.shape_buff.bbox, self.layout.container.outer_cd.bbox)
            } else {
                0.0
            };
            SampleEval::Clear { loss }
        } else {
            // Some collisions detected but withing the upper bound, return collision with total loss
            SampleEval::Collision {
                loss: self.collector.loss(&self.shape_buff),
            }
        }
    }

    fn n_evals(&self) -> usize {
        self.n_evals
    }
}


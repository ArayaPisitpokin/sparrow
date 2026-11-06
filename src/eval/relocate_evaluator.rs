use crate::eval::sample_eval::{SampleEval, SampleEvaluator};
use crate::small::positional_loss;
use jagua_rs::collision_detection::hazards::HazKey;
use jagua_rs::entities::{Item, Layout, PItemKey};
use jagua_rs::geometry::geo_traits::TransformableFrom;
use jagua_rs::geometry::primitives::SPolygon;
use jagua_rs::geometry::DTransformation;

/// Evaluator for the small-piece relocation operator (`optimizer::small_relocate`):
/// like [`crate::eval::lbf_evaluator::LBFEvaluator`] it rejects any colliding sample,
/// but it ignores the hazard of the item's *own current placement* (the item stays in
/// the layout while a new spot is searched) and ranks clear samples by the signed
/// positional objective of `crate::small`.
pub struct RelocateEvaluator<'a> {
    layout: &'a Layout,
    item: &'a Item,
    /// Hazard key of the item's current placement — deemed irrelevant during the search.
    ignore: HazKey,
    bias: f32,
    shape_buff: SPolygon,
    n_evals: usize,
}

impl<'a> RelocateEvaluator<'a> {
    /// `bias` follows `SmallPieceConfig::clear_bias`; 0 falls back to leftmost (+1).
    pub fn new(layout: &'a Layout, item: &'a Item, current_pk: PItemKey, bias: f32) -> Self {
        let ignore = layout
            .cde()
            .haz_key_from_pi_key(current_pk)
            .expect("placed item should be registered in the CDE");
        Self {
            layout,
            item,
            ignore,
            bias: if bias == 0.0 { 1.0 } else { bias },
            shape_buff: item.shape_cd.as_ref().clone(),
            n_evals: 0,
        }
    }
}

impl<'a> SampleEvaluator for RelocateEvaluator<'a> {
    fn evaluate_sample(&mut self, dt: DTransformation, _upper_bound: Option<SampleEval>) -> SampleEval {
        self.n_evals += 1;
        let cde = self.layout.cde();
        let transf = dt.into();
        if cde.detect_surrogate_collision(self.item.shape_cd.surrogate(), &transf, &self.ignore) {
            return SampleEval::Invalid;
        }
        self.shape_buff.transform_from(&self.item.shape_cd, &transf);
        if cde.detect_poly_collision(&self.shape_buff, &self.ignore) {
            return SampleEval::Invalid;
        }
        SampleEval::Clear {
            loss: positional_loss(self.bias, self.shape_buff.bbox, self.layout.container.outer_cd.bbox),
        }
    }

    fn n_evals(&self) -> usize {
        self.n_evals
    }
}

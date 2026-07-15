//! Grouped-orientation ("nap-up-and-down") support.
//!
//! A `GroupedOrientationSpec` declares that the instance's items form garment groups:
//! all pieces of one physical garment must share an orientation (0° or 180°), while
//! different garments may be oriented independently. Items are expected to carry
//! `allowed_orientations = [0, 180]`; the per-garment coupling is enforced as a search
//! invariant (see `docs/grouped_orientation_design.md` in the consuming repo):
//!
//! * the initial solution encodes a valid partition,
//! * every regular move preserves the moved copy's rotation
//!   (`SeparatorConfig::preserve_rotations`),
//! * only the group-flip disruption changes rotations, one whole garment at a time.
//!
//! With `None` passed everywhere, nothing in this module executes.

use jagua_rs::probs::spp::entities::SPInstance;

/// Side-channel description of the garment groups of an instance.
#[derive(Debug, Clone)]
pub struct GroupedOrientationSpec {
    /// One entry per garment *type*. Item ids must be disjoint across groups.
    pub groups: Vec<OrientationGroup>,
    /// Configuration of the group-flip disruption move.
    pub flip: GroupFlipConfig,
}

/// One garment type: its piece-type items and how many garments of it exist.
#[derive(Debug, Clone)]
pub struct OrientationGroup {
    /// `(item_id, per_garment)` — one entry per piece type of this garment type.
    pub members: Vec<(usize, usize)>,
    /// Number of garments of this type in the marker.
    pub quantity: usize,
}

/// Tuning knobs for the group-flip disruption.
#[derive(Debug, Clone, Copy)]
pub struct GroupFlipConfig {
    /// Probability that a disruption event is a group flip instead of the
    /// swap-two-large-items move. `0.0` disables flips entirely (strict `<` compare).
    pub p_flip: f32,
    /// Number of garments flipped per flip event.
    pub n_garments_per_flip: usize,
}

impl GroupedOrientationSpec {
    /// Validates the spec against an instance. Called once per `optimize()`;
    /// not on any hot path.
    pub fn validate(&self, instance: &SPInstance) -> Result<(), String> {
        if !(0.0..=1.0).contains(&self.flip.p_flip) {
            return Err(format!("p_flip must be in [0, 1], got {}", self.flip.p_flip));
        }
        if self.flip.n_garments_per_flip == 0 {
            return Err("n_garments_per_flip must be >= 1".to_string());
        }
        if self.groups.is_empty() {
            return Err("spec contains no groups".to_string());
        }
        let n_items = instance.items.len();
        let mut seen = vec![false; n_items];
        for (g_idx, g) in self.groups.iter().enumerate() {
            if g.members.is_empty() {
                return Err(format!("group {g_idx} has no members"));
            }
            if g.quantity == 0 {
                return Err(format!("group {g_idx} has quantity 0"));
            }
            for &(item_id, per_garment) in &g.members {
                if item_id >= n_items {
                    return Err(format!(
                        "group {g_idx}: item id {item_id} out of range (instance has {n_items} items)"
                    ));
                }
                if seen[item_id] {
                    return Err(format!(
                        "item id {item_id} appears in more than one group (groups must be disjoint)"
                    ));
                }
                seen[item_id] = true;
                if per_garment == 0 {
                    return Err(format!("group {g_idx}: item {item_id} has per_garment 0"));
                }
                let demand = instance.item_qty(item_id);
                if demand != g.quantity * per_garment {
                    return Err(format!(
                        "group {g_idx}: item {item_id} demand {demand} != quantity {} x per_garment {per_garment}",
                        g.quantity
                    ));
                }
            }
        }
        Ok(())
    }
}

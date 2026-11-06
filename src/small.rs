//! Small-piece mechanisms for bimodal-size instances (few large "panels" plus many
//! small pieces), following the 2026-08-28 diagnostic in the consuming repo
//! (`docs/research/small_piece_mechanism_2026-08-28.md`).
//!
//! Two of the separator's founding assumptions — only colliding items move, and every
//! collision-free position is equally good — hold for homogeneous instances but turn
//! the small pieces of a bimodal instance into randomly scattered obstacles inside the
//! panel skeleton: they relocate 5–13x more often than panels, never settle at holes
//! or the frontier, and ~45% of failed separations involve *only* small pieces yet are
//! answered by the swap-two-large-items disruption. Three flag-gated mechanisms, all
//! off by default (`SmallPieceConfig::OFF` reproduces upstream behaviour exactly):
//!
//! * **A** — positional tiebreak among the collision-free samples of a small item
//!   (`clear_bias`, signed: > 0 prefers the leftmost clear spot, < 0 the frontier);
//! * **M** — voluntary move turns for collision-free small items (`clear_move_prob`),
//!   so smalls seek holes/frontier without waiting to be bumped (requires A);
//! * **D** — when a separation fails with small pieces among the colliders, re-place
//!   those pieces with a dense clear-position search before the swap-two-large-items
//!   disruption is considered (`relocate_mode`, see `optimizer::small_relocate`);
//! * **T** — a tidy pass on the master state before every separation: colliding small
//!   pieces are re-placed, and optionally every clear small piece migrates to a better
//!   positional spot (`tidy_mode`; this is M applied where it persists).

use jagua_rs::geometry::primitives::Rect;
use jagua_rs::probs::spp::entities::SPInstance;
use std::sync::Arc;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SmallPieceConfig {
    /// "Large" items are those whose convex-hull area reaches this cumulative share of
    /// the total (largest first) — the same rule the swap disruption uses to pick its
    /// candidates; every other item is "small".
    pub cutoff_percentile: f32,
    /// A: signed positional bias for collision-free samples of small items. 0 disables;
    /// > 0 prefers smaller x (leftmost), < 0 prefers larger x (the frontier). Also the
    /// objective of D's relocation search (D falls back to leftmost when this is 0).
    pub clear_bias: f32,
    /// M: per-pass probability that a collision-free small item gets a move turn.
    /// Inert unless `clear_bias != 0` (a voluntary move needs an objective).
    pub clear_move_prob: f32,
    /// D: after a failed separation, re-place the colliding small pieces with a dense
    /// clear-position search. 0 = off; 1 = only when *every* collider is small;
    /// 2 = whenever at least one collider is small (panels are left to the swap, which
    /// is then correctly diagnosed).
    pub relocate_mode: u8,
    /// D: container samples per relocated piece (LBF-style dense search).
    pub relocate_samples: usize,
    /// T: tidy pass on the master state right before every separation. 0 = off;
    /// 1 = re-place the *colliding* small pieces (D applied before the GLS wastes its
    /// strikes); 2 = additionally migrate every collision-free small piece to a clear
    /// spot with a better positional loss (mechanism M, applied where it persists —
    /// in-separation voluntary moves are rolled back at every strike).
    pub tidy_mode: u8,
    /// T: container samples per piece in the tidy pass.
    pub tidy_samples: usize,
    /// A: extra container samples for small items in regular moves (their evaluations
    /// are cheap; raises the odds that a clear spot is offered at all).
    pub extra_container_samples: usize,
    /// N: after a failed separation whose colliders include small pieces, push the
    /// panels bordering the jam outward so the GLS can re-negotiate the pockets — a
    /// local, structure-preserving alternative to swapping two large items across the
    /// strip. A "small-only" failure is a jam: the smalls collide only with each other
    /// or the wall inside pockets that are collectively too small, while every
    /// bordering panel is clear and therefore never moves. 0 = off; 1 = small-only
    /// failures; 2 = any failure with small colliders.
    pub nudge_mode: u8,
    /// N: push distance as a fraction of the mean diameter of the colliding small pieces.
    pub nudge_step_ratio: f32,
    /// N: a panel borders the jam when its bounding box lies within this fraction of the
    /// mean colliding-small diameter of a colliding small piece's bounding box.
    pub nudge_radius_ratio: f32,
    /// N: consecutive nudges allowed before the swap disruption takes over.
    pub nudge_max: u8,
    /// Y (T1): panel-yield inside `separate()`. When the inner separation loop
    /// stalls while only small pieces are colliding (a jam), the clear panels
    /// bordering the jam are stepped outward to the nearest *collision-free*
    /// position along the outward direction — a loss-neutral move applied to the
    /// master state between iterations. It survives the strike rollback exactly
    /// when a subsequent small-escape lowers the loss (the min-loss snapshot then
    /// contains the yielded panel), integrating panel yielding into the GLS
    /// acceptance instead of bolting it on outside (mechanism N's flaw).
    /// 0 = off; 1 = on.
    pub yield_mode: u8,
    /// Y: inner-loop iterations without improvement before a yield volley fires
    /// (and again every multiple thereof, until `yield_max`).
    pub yield_stall: usize,
    /// Y: maximum outward displacement as a fraction of the mean diameter of the
    /// colliding small pieces (candidates at 1.0/0.7/0.45/0.25 x this, nearest
    /// clear one wins).
    pub yield_step_ratio: f32,
    /// Y: a panel borders the jam when its bbox lies within this fraction of the
    /// mean colliding-small diameter of a colliding small piece's bbox.
    pub yield_radius_ratio: f32,
    /// Y: maximum yield volleys per `separate()` call.
    pub yield_max: usize,
    /// Y: jam gate tolerance — a volley may fire while at most this many big pieces
    /// are colliding (0 = strict small-only; mid-loop states almost always carry a
    /// transient big collision somewhere, which starves the strict gate).
    pub yield_gate_bigs: u8,
    /// L: lane discipline for large pieces. Every sample of a large item pays
    /// `lane_weight x (normalized distance of its centroid-y to the nearest lane
    /// centre) x sqrt(convex-hull area)` — the panel-side analogue of the anneal
    /// pressure, nudging panels into the horizontal lanes a production marker uses.
    /// 0 disables.
    pub lane_weight: f32,
    /// L: number of lanes; 0 = derive as round(strip height / median large-item
    /// bbox height), clamped to [1, 6].
    pub lane_count: u8,
}

impl SmallPieceConfig {
    /// Upstream behaviour: no mechanism active.
    pub const OFF: Self = Self {
        cutoff_percentile: 0.75,
        clear_bias: 0.0,
        clear_move_prob: 0.0,
        relocate_mode: 0,
        relocate_samples: 1000,
        tidy_mode: 0,
        tidy_samples: 300,
        extra_container_samples: 0,
        nudge_mode: 0,
        nudge_step_ratio: 0.5,
        nudge_radius_ratio: 1.0,
        nudge_max: 1,
        yield_mode: 0,
        yield_stall: 6,
        yield_step_ratio: 0.5,
        yield_radius_ratio: 1.5,
        yield_max: 8,
        yield_gate_bigs: 0,
        lane_weight: 0.0,
        lane_count: 0,
    };

    pub fn relocate_enabled(&self) -> bool {
        self.relocate_mode > 0
    }

    pub fn voluntary_moves_active(&self) -> bool {
        self.clear_move_prob > 0.0 && self.clear_bias != 0.0
    }

    pub fn any_enabled(&self) -> bool {
        self.clear_bias != 0.0
            || self.voluntary_moves_active()
            || self.relocate_enabled()
            || self.tidy_mode > 0
            || self.extra_container_samples > 0
            || self.nudge_mode > 0
            || self.lane_weight > 0.0
    }
}

/// item_id -> whether the item type is "small": its convex-hull area lies below the
/// area of the item at which the largest-first cumulative convex-hull area (weighted by
/// demand) crosses `cutoff_percentile` of the total. Mirrors the swap disruption's
/// "large" definition (`explore::disrupt_solution`), so the two classifications agree.
pub fn small_item_mask(instance: &SPInstance, cutoff_percentile: f32) -> Arc<[bool]> {
    let ch_area = |id: usize| instance.items[id].0.shape_cd.surrogate().convex_hull_area;
    let total: f32 = instance
        .items
        .iter()
        .map(|(item, qty)| item.shape_cd.surrogate().convex_hull_area * *qty as f32)
        .sum();
    let threshold = total * cutoff_percentile;

    let mut order: Vec<usize> = (0..instance.items.len()).collect();
    order.sort_by(|&a, &b| ch_area(b).total_cmp(&ch_area(a)));

    let mut cumulative = 0.0;
    let mut cutoff = 0.0;
    for id in order {
        cumulative += ch_area(id) * instance.items[id].1 as f32;
        if cumulative > threshold {
            cutoff = ch_area(id);
            break;
        }
    }
    (0..instance.items.len()).map(|id| ch_area(id) < cutoff).collect()
}

/// Lane centres for `lane_count` horizontal lanes (0 = auto: strip height over the
/// median large-item bbox height, clamped to [1, 6]) across a strip of height `h`.
pub fn lane_centers(lane_count: u8, h: f32, median_big_height: f32) -> Vec<f32> {
    let k = if lane_count > 0 {
        lane_count as usize
    } else {
        ((h / median_big_height.max(1.0)).round() as usize).clamp(1, 6)
    };
    (0..k).map(|i| h * (2 * i + 1) as f32 / (2 * k) as f32).collect()
}

/// Positional loss of a collision-free placement: distance (in strip units) from the
/// preferred end of the strip, scaled by |bias|. Non-negative, zero at the preferred wall.
#[inline]
pub fn positional_loss(bias: f32, shape_bbox: Rect, container_bbox: Rect) -> f32 {
    if bias > 0.0 {
        bias * (shape_bbox.x_min - container_bbox.x_min).max(0.0)
    } else {
        -bias * (container_bbox.x_max - shape_bbox.x_max).max(0.0)
    }
}

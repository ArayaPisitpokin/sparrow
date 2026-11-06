//! Small-piece mechanism D (see `crate::small`): after a failed separation, classify
//! the colliding pieces by size class and — when only small pieces are colliding —
//! re-place each of them with a dense clear-position search instead of handing the
//! layout to the swap-two-large-items disruption. Runs on the master thread between
//! separations; never on a hot path.

use crate::eval::relocate_evaluator::RelocateEvaluator;
use crate::eval::sample_eval::{SampleEval, SampleEvaluator};
use crate::optimizer::separator::Separator;
use crate::sample::search::{search_placement, SampleConfig};
use crate::small::positional_loss;
use crate::util::listener::SearchStats;
use jagua_rs::entities::{Instance, PItemKey};
use jagua_rs::geometry::primitives::Rect;
use jagua_rs::geometry::DTransformation;
use log::{debug, info};
use std::sync::atomic::Ordering;

/// Size-class composition of the colliding pieces of a layout.
#[derive(Debug, Clone, Copy, Default)]
pub struct ColliderCensus {
    pub n_small: usize,
    pub n_big: usize,
}

impl ColliderCensus {
    pub fn small_only(&self) -> bool {
        self.n_small > 0 && self.n_big == 0
    }

    /// Counts this failed separation into the phase statistics.
    pub fn record(&self, stats: &mut SearchStats) {
        match (self.n_small > 0, self.n_big > 0) {
            (true, false) => stats.n_fail_small_only += 1,
            (true, true) => stats.n_fail_mixed += 1,
            (false, true) => stats.n_fail_big_only += 1,
            (false, false) => {}
        }
    }
}

/// Who is colliding in the separator's current layout.
pub fn classify_colliders(sep: &Separator) -> ColliderCensus {
    let mut census = ColliderCensus::default();
    for (pk, pi) in sep.prob.layout.placed_items.iter() {
        if sep.ct.get_loss(pk) > 0.0 {
            if sep.small_items[pi.item_id] {
                census.n_small += 1;
            } else {
                census.n_big += 1;
            }
        }
    }
    census
}

/// Searches a collision-free spot for the placed piece `pk` (ignoring its own current
/// placement) with the signed positional objective; returns the best clear pose and
/// its positional loss, plus the evaluations spent.
fn search_clear_spot(
    sep: &mut Separator,
    pk: PItemKey,
    n_samples: usize,
    bias: f32,
) -> (Option<(jagua_rs::geometry::DTransformation, f32)>, usize) {
    let item_id = sep.prob.layout.placed_items[pk].item_id;
    // Grouped-orientation locks apply here exactly as in regular moves.
    let rot_lock = match sep.locked_items.as_deref() {
        Some(locked) if locked[item_id] && sep.locks_active.load(Ordering::Relaxed) => {
            Some(sep.prob.layout.placed_items[pk].d_transf.rotation())
        }
        _ => None,
    };
    let sample_config = SampleConfig {
        n_container_samples: n_samples.max(1),
        n_focussed_samples: 0,
        n_coord_descents: 3,
        window_home: None,
    };
    let item = sep.instance.item(item_id);
    let evaluator = RelocateEvaluator::new(&sep.prob.layout, item, pk, bias);
    let (best, evals) = search_placement(
        &sep.prob.layout,
        item,
        None,
        evaluator,
        sample_config,
        &mut sep.rng,
        rot_lock,
    );
    match best {
        Some((dt, SampleEval::Clear { loss })) => (Some((dt, loss)), evals),
        _ => (None, evals),
    }
}

/// The D operator. Fires when the colliding set is all-small (mode 1) or contains at
/// least one small piece (mode 2); relocates the *small* colliders only. Returns the
/// number of pieces that found a collision-free spot (`None` when it did not fire).
/// Pieces are processed largest first; a piece freed by an earlier relocation is
/// skipped. Every successful relocation strictly lowers the total loss, so no
/// rollback is needed.
pub fn relocate_colliding_smalls(
    sep: &mut Separator,
    census: ColliderCensus,
    stats: &mut SearchStats,
) -> Option<usize> {
    let mode = sep.config.small.relocate_mode;
    let fires = census.n_small > 0 && (mode == 2 || (mode == 1 && census.n_big == 0));
    if !fires {
        return None;
    }
    stats.n_small_relocate_events += 1;
    let loss_before = sep.ct.get_total_loss();

    let mut colliding: Vec<(PItemKey, f32)> = sep
        .prob
        .layout
        .placed_items
        .iter()
        .filter(|(pk, pi)| sep.small_items[pi.item_id] && sep.ct.get_loss(*pk) > 0.0)
        .map(|(pk, pi)| (pk, pi.shape.area))
        .collect();
    colliding.sort_by(|a, b| b.1.total_cmp(&a.1));

    let n_samples = sep.config.small.relocate_samples;
    let bias = sep.config.small.clear_bias;
    let mut n_relocated = 0usize;
    let mut n_evals = 0usize;
    for (pk, _) in colliding {
        if sep.ct.get_loss(pk) == 0.0 {
            continue; // freed when a partner moved away
        }
        let (best, evals) = search_clear_spot(sep, pk, n_samples, bias);
        n_evals += evals;
        if let Some((dt, _)) = best {
            sep.move_item(pk, dt);
            n_relocated += 1;
        }
    }

    sep.cum_evals += n_evals as u64;
    sep.cum_moves += n_relocated as u64;
    stats.n_small_pieces_relocated += n_relocated as u64;
    let loss_after = sep.ct.get_total_loss();
    if loss_after == 0.0 {
        stats.n_small_relocate_success += 1;
    }
    info!(
        "[SMALL] relocated {n_relocated}/{} small colliders, loss {loss_before:.3} -> {loss_after:.3}{}",
        census.n_small,
        if loss_after == 0.0 { " (feasible)" } else { "" }
    );
    Some(n_relocated)
}

/// Euclidean gap between two axis-aligned rectangles (0 when they overlap).
fn rect_gap(a: Rect, b: Rect) -> f32 {
    let gx = (b.x_min - a.x_max).max(a.x_min - b.x_max).max(0.0);
    let gy = (b.y_min - a.y_max).max(a.y_min - b.y_max).max(0.0);
    (gx * gx + gy * gy).sqrt()
}

/// The N operator: pushes every panel bordering a jam of colliding small pieces
/// outward (away from the jammed pieces' centroids) by `nudge_step_ratio` x the mean
/// colliding-small diameter, keeping rotations. The following separation repairs the
/// induced panel collisions — a coherent outward push the GLS propagates through the
/// skeleton, unlike the swap's relocation of two panels across the strip. Returns the
/// number of panels moved.
pub fn nudge_panels_around_jam(sep: &mut Separator, stats: &mut SearchStats) -> usize {
    let cfg = sep.config.small;
    let jams: Vec<((f32, f32), Rect, f32)> = sep
        .prob
        .layout
        .placed_items
        .iter()
        .filter(|(pk, pi)| sep.small_items[pi.item_id] && sep.ct.get_loss(*pk) > 0.0)
        .map(|(_, pi)| {
            let c = pi.shape.centroid();
            ((c.0, c.1), pi.shape.bbox, pi.shape.diameter)
        })
        .collect();
    if jams.is_empty() {
        return 0;
    }
    let mean_diam = jams.iter().map(|j| j.2).sum::<f32>() / jams.len() as f32;
    let step = cfg.nudge_step_ratio * mean_diam;
    let radius = cfg.nudge_radius_ratio * mean_diam;

    let mut moves: Vec<(PItemKey, DTransformation)> = Vec::new();
    for (pk, pi) in sep.prob.layout.placed_items.iter() {
        if sep.small_items[pi.item_id] {
            continue;
        }
        let (mut dx, mut dy, mut n) = (0.0f32, 0.0f32, 0usize);
        for ((jx, jy), jb, _) in &jams {
            if rect_gap(pi.shape.bbox, *jb) <= radius {
                let c = pi.shape.centroid();
                let (vx, vy) = (c.0 - jx, c.1 - jy);
                let len = (vx * vx + vy * vy).sqrt().max(1e-3);
                dx += vx / len;
                dy += vy / len;
                n += 1;
            }
        }
        if n > 0 {
            let len = (dx * dx + dy * dy).sqrt().max(1e-3);
            let t = pi.d_transf.translation();
            moves.push((
                pk,
                DTransformation::new(pi.d_transf.rotation(), (t.0 + step * dx / len, t.1 + step * dy / len)),
            ));
        }
    }
    for (pk, dt) in &moves {
        sep.move_item(*pk, *dt);
    }
    stats.n_nudges += 1;
    stats.n_panels_nudged += moves.len() as u64;
    info!(
        "[SMALL] nudge: {} jammed smalls (mean diam {mean_diam:.1}), pushed {} bordering panels by {step:.1}, loss now {:.3}",
        jams.len(),
        moves.len(),
        sep.ct.get_total_loss()
    );
    moves.len()
}

/// The T operator: a tidy pass on the master state right before a separation.
/// Colliding small pieces are re-placed into clear spots (mode >= 1); with
/// `migrate_clear` every collision-free small piece is additionally moved to a clear
/// spot with a strictly better positional loss (mode 2). Only clear destinations are
/// ever taken, so the total loss never increases. Returns the number of pieces moved.
pub fn tidy_smalls(sep: &mut Separator, stats: &mut SearchStats, migrate_clear: bool) -> usize {
    stats.n_tidy_calls += 1;
    let bias = sep.config.small.clear_bias;
    let n_samples = sep.config.small.tidy_samples;
    let container_bbox = sep.prob.layout.container.outer_cd.bbox;

    // candidates: small pieces, colliding first, then (optionally) the clear ones; largest first
    let mut cands: Vec<(PItemKey, bool, f32)> = sep
        .prob
        .layout
        .placed_items
        .iter()
        .filter(|(_, pi)| sep.small_items[pi.item_id])
        .map(|(pk, pi)| (pk, sep.ct.get_loss(pk) > 0.0, pi.shape.area))
        .filter(|(_, colliding, _)| *colliding || migrate_clear)
        .collect();
    cands.sort_by(|a, b| b.1.cmp(&a.1).then(b.2.total_cmp(&a.2)));

    let mut n_moved = 0usize;
    let mut n_evals = 0usize;
    for (pk, was_colliding, _) in cands {
        let colliding_now = sep.ct.get_loss(pk) > 0.0;
        if was_colliding && !colliding_now {
            continue; // freed by an earlier move
        }
        let (best, evals) = search_clear_spot(sep, pk, n_samples, bias);
        n_evals += evals;
        let Some((dt, new_pos_loss)) = best else { continue };
        if !colliding_now {
            // a clear piece moves only to a strictly better positional loss
            let b = if bias == 0.0 { 1.0 } else { bias };
            let cur = positional_loss(b, sep.prob.layout.placed_items[pk].shape.bbox, container_bbox);
            if new_pos_loss >= cur - 1e-3 {
                continue;
            }
        }
        sep.move_item(pk, dt);
        n_moved += 1;
    }
    sep.cum_evals += n_evals as u64;
    sep.cum_moves += n_moved as u64;
    stats.n_tidy_moves += n_moved as u64;
    debug!("[SMALL] tidy: moved {n_moved} small pieces ({n_evals} evals)");
    n_moved
}

/// The Y operator (T1, 2026-08-31): panel-yield *inside* `separate()`. Called from
/// the separator's inner loop on a stall while only small pieces are colliding (a
/// jam — see the 08-28 mechanism note: the smalls have no clear spot anywhere and
/// every bordering panel is collision-free, hence inert). Each clear panel bordering
/// the jam is stepped outward to the nearest collision-free position along the
/// outward direction (candidates at 1.0/0.7/0.45/0.25 x the step; nearest wins).
/// Loss-neutral by construction (clear -> clear), applied to the master state; it
/// survives the strike rollback exactly when a subsequent small-escape lowers the
/// loss, so the yield participates in the GLS acceptance instead of being erased by
/// it (mechanism N's flaw). Returns the number of panels moved.
pub fn yield_panels_in_sep(sep: &mut Separator) -> usize {
    let cfg = sep.config.small;
    let jams: Vec<((f32, f32), Rect, f32)> = sep
        .prob
        .layout
        .placed_items
        .iter()
        .filter(|(pk, pi)| sep.small_items[pi.item_id] && sep.ct.get_loss(*pk) > 0.0)
        .map(|(_, pi)| {
            let c = pi.shape.centroid();
            ((c.0, c.1), pi.shape.bbox, pi.shape.diameter)
        })
        .collect();
    if jams.is_empty() {
        return 0;
    }
    let mean_diam = jams.iter().map(|j| j.2).sum::<f32>() / jams.len() as f32;
    let step = cfg.yield_step_ratio * mean_diam;
    let radius = cfg.yield_radius_ratio * mean_diam;

    // Clear panels bordering a jam, with their unit outward direction.
    let cands: Vec<(PItemKey, f32, (f32, f32), (f32, f32))> = sep
        .prob
        .layout
        .placed_items
        .iter()
        .filter(|(pk, pi)| !sep.small_items[pi.item_id] && sep.ct.get_loss(*pk) == 0.0)
        .filter_map(|(pk, pi)| {
            let (mut dx, mut dy, mut n) = (0.0f32, 0.0f32, 0usize);
            for ((jx, jy), jb, _) in &jams {
                if rect_gap(pi.shape.bbox, *jb) <= radius {
                    let c = pi.shape.centroid();
                    let (vx, vy) = (c.0 - jx, c.1 - jy);
                    let len = (vx * vx + vy * vy).sqrt().max(1e-3);
                    dx += vx / len;
                    dy += vy / len;
                    n += 1;
                }
            }
            if n == 0 {
                return None;
            }
            let len = (dx * dx + dy * dy).sqrt().max(1e-3);
            let t = pi.d_transf.translation();
            Some((
                pk,
                pi.d_transf.rotation(),
                (t.0, t.1),
                (dx / len, dy / len),
            ))
        })
        .collect();

    let mut n_moved = 0usize;
    for (pk, rot, (tx, ty), (dx, dy)) in cands {
        // pk stays valid: only *this* item's key changes on its own move.
        let item_id = sep.prob.layout.placed_items[pk].item_id;
        let chosen = {
            let item = sep.instance.item(item_id);
            let mut evaluator =
                RelocateEvaluator::new(&sep.prob.layout, item, pk, 1.0);
            // Fan of candidates: farthest clear position wins (max opening); the
            // outward ray first, then +-25 and +-50 degree deflections, at
            // descending radii down to ~6% of the step (a wedged panel usually
            // has no clear spot on the pure outward ray — a small slide still
            // opens the pocket).
            const DEFLECT: [f32; 5] = [0.0, 0.4363, -0.4363, 0.8727, -0.8727];
            [1.0f32, 0.7, 0.45, 0.25, 0.12, 0.06]
                .iter()
                .find_map(|frac| {
                    let d = frac * step;
                    DEFLECT.iter().find_map(|a| {
                        let (sa, ca) = a.sin_cos();
                        let (rx, ry) = (dx * ca - dy * sa, dx * sa + dy * ca);
                        let dt = DTransformation::new(rot, (tx + d * rx, ty + d * ry));
                        match evaluator.evaluate_sample(dt, None) {
                            SampleEval::Clear { .. } => Some(dt),
                            _ => None,
                        }
                    })
                })
        };
        if let Some(dt) = chosen {
            sep.move_item(pk, dt);
            n_moved += 1;
        }
    }
    debug!(
        "[SMALL] yield: {} jammed smalls, stepped {} bordering panels (step {step:.1}), loss {:.3}",
        jams.len(),
        n_moved,
        sep.ct.get_total_loss()
    );
    n_moved
}

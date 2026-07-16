//! Soft-anneal ("anneal" mode): search in the relaxed orientation space with a
//! growing group-consistency pressure, project onto the constraint at the end.
//!
//! During the flip window, pieces sample both orientation classes freely (their
//! rotation locks are inactive), while every candidate pose whose class
//! disagrees with its garment's *current majority* pays a penalty λ(t) scaled
//! to the piece's own collision magnitude. λ ramps quadratically from 0 to
//! `anneal_lambda` over the window: early, layout quality dominates and the
//! pieces organize freely (the regime where the free-mode search excels);
//! late, consistency dominates and garments coalesce while they still have
//! room to comply. At window close the few remaining minority pieces are
//! flipped in place onto their garment's majority (the projection), rotation
//! locks activate, and the remaining budget refines a now-valid layout.
//!
//! Garment identity during the anneal is provisional: copies are re-clustered
//! spatially every refresh (anchor-member chunks by x, remaining members
//! assigned nearest-with-capacity), so the pressure always points at the
//! *current* geometry, not a stale assignment.

use crate::grouped::{is_dir_180, GroupedOrientationSpec};
use crate::optimizer::separator::Separator;
use crate::util::listener::SearchStats;
use jagua_rs::entities::{Instance, PItemKey};
use jagua_rs::geometry::DTransformation;
use jagua_rs::geometry::geo_enums::RotationRange;
use log::info;
use slotmap::SecondaryMap;

/// Per-copy anneal data, rebuilt by the master before each parallel move pass.
/// `penalty[c]` is added to any candidate pose of this copy in class `c`
/// (0 = the 0° class, 1 = the 180° class); `move_candidate` marks copies that
/// currently disagree with their garment majority, so they get a move turn
/// even while collision-free (pressure must bind on clear pieces too).
#[derive(Debug, Clone, Copy, Default)]
pub struct AnnealEntry {
    pub penalty: [f32; 2],
    pub move_candidate: bool,
}

pub type AnnealMap = SecondaryMap<PItemKey, AnnealEntry>;

/// Clusters the group members' copies into physical garments at the current
/// geometry, ignoring orientation classes (valid on mid-anneal mixed layouts).
/// Returns one Vec<PItemKey> per garment cell.
fn cluster_cells(sep: &Separator, spec: &GroupedOrientationSpec) -> Vec<Vec<PItemKey>> {
    let n_items = sep.instance.items.len();
    let mut copies: Vec<Vec<(PItemKey, (f32, f32))>> = vec![Vec::new(); n_items];
    for (pk, pi) in sep.prob.layout.placed_items.iter() {
        let c = pi.shape.centroid();
        copies[pi.item_id].push((pk, (c.0, c.1)));
    }

    let mut cells: Vec<Vec<PItemKey>> = Vec::new();
    for group in &spec.groups {
        let &(anchor_id, anchor_pg) = group
            .members
            .iter()
            .min_by(|a, b| {
                a.1.cmp(&b.1).then(
                    sep.instance
                        .item(b.0)
                        .shape_cd
                        .area
                        .total_cmp(&sep.instance.item(a.0).shape_cd.area),
                )
            })
            .expect("groups have >= 1 member (validated)");
        let mut anchors = copies[anchor_id].clone();
        if anchors.is_empty() {
            continue;
        }
        anchors.sort_by(|a, b| a.1 .0.total_cmp(&b.1 .0));
        let n_cells = anchors.len() / anchor_pg;
        let mut cell_centers: Vec<(f32, f32)> = Vec::with_capacity(n_cells);
        let mut cell_pks: Vec<Vec<PItemKey>> = Vec::with_capacity(n_cells);
        for chunk in anchors.chunks(anchor_pg) {
            let cx = chunk.iter().map(|c| c.1 .0).sum::<f32>() / chunk.len() as f32;
            let cy = chunk.iter().map(|c| c.1 .1).sum::<f32>() / chunk.len() as f32;
            cell_centers.push((cx, cy));
            cell_pks.push(chunk.iter().map(|c| c.0).collect());
        }
        for &(item_id, pg) in &group.members {
            if item_id == anchor_id {
                continue;
            }
            let mut cls = copies[item_id].clone();
            cls.sort_by(|a, b| a.1 .0.total_cmp(&b.1 .0));
            let mut cap = vec![pg; n_cells];
            for (pk, c) in cls {
                let Some(best) = (0..n_cells).filter(|&i| cap[i] > 0).min_by(|&i, &j| {
                    d2(c, cell_centers[i]).total_cmp(&d2(c, cell_centers[j]))
                }) else {
                    continue; // capacity exhausted (shouldn't happen; degrade gracefully)
                };
                cap[best] -= 1;
                cell_pks[best].push(pk);
            }
        }
        cells.extend(cell_pks);
    }
    cells
}

/// Builds the per-copy penalty map for the current geometry and λ.
/// Penalty unit = the piece's convex-hull area (its own collision magnitude
/// scale under the `quantify` family), times λ. Cells with a class tie apply
/// no pressure this round.
pub(crate) fn build_anneal_map(
    sep: &Separator,
    spec: &GroupedOrientationSpec,
    lambda: f32,
) -> AnnealMap {
    let mut map = AnnealMap::new();
    if lambda <= 0.0 {
        return map;
    }
    for cell in cluster_cells(sep, spec) {
        // area-weighted class vote of the cell
        let mut vote = [0.0f32; 2];
        for &pk in &cell {
            let pi = &sep.prob.layout.placed_items[pk];
            let class = is_dir_180(pi.d_transf.rotation()) as usize;
            vote[class] += pi.shape.surrogate().convex_hull_area;
        }
        if (vote[0] - vote[1]).abs() <= f32::EPSILON * (vote[0] + vote[1]) {
            continue; // tie: no pressure
        }
        let majority = (vote[1] > vote[0]) as usize;
        for &pk in &cell {
            let pi = &sep.prob.layout.placed_items[pk];
            let unit = pi.shape.surrogate().convex_hull_area;
            let mut penalty = [0.0f32; 2];
            penalty[1 - majority] = lambda * unit;
            let current = is_dir_180(pi.d_transf.rotation()) as usize;
            map.insert(
                pk,
                AnnealEntry {
                    penalty,
                    move_candidate: current != majority,
                },
            );
        }
    }
    map
}

/// Projects the layout onto full group consistency: every cell's minority
/// pieces are flipped in place onto the cell majority. Returns the number of
/// pieces flipped. After this the group counting invariant holds and rotation
/// locks may activate.
pub(crate) fn project_to_consistency(
    sep: &mut Separator,
    spec: &GroupedOrientationSpec,
    stats: &mut SearchStats,
) -> usize {
    let cells = cluster_cells(sep, spec);
    let mut moves: Vec<(PItemKey, DTransformation)> = Vec::new();
    for cell in &cells {
        let mut vote = [0.0f32; 2];
        for &pk in cell {
            let pi = &sep.prob.layout.placed_items[pk];
            vote[is_dir_180(pi.d_transf.rotation()) as usize] +=
                pi.shape.surrogate().convex_hull_area;
        }
        // ties resolve to the 0° class — any uniform choice is valid
        let majority180 = vote[1] > vote[0];
        for &pk in cell {
            let pi = &sep.prob.layout.placed_items[pk];
            let cur180 = is_dir_180(pi.d_transf.rotation());
            if cur180 == majority180 {
                continue;
            }
            let item = sep.instance.item(pi.item_id);
            let RotationRange::Discrete(allowed) = &item.allowed_rotation else {
                continue;
            };
            let Some(target) = allowed.iter().copied().find(|r| is_dir_180(*r) == majority180)
            else {
                continue;
            };
            let c = pi.shape.centroid();
            let t = pi.d_transf.translation();
            moves.push((
                pk,
                DTransformation::new(target, (2.0 * c.0 - t.0, 2.0 * c.1 - t.1)),
            ));
        }
    }
    let n = moves.len();
    let mut loss_injected = 0.0;
    for (pk, dt) in moves {
        let new_pk = sep.move_item(pk, dt);
        loss_injected += sep.ct.get_loss(new_pk);
    }
    info!(
        "[ANNL] projected onto consistency: {n} pieces flipped across {} cells, loss injected {loss_injected:.3}",
        cells.len(),
    );
    if n > 0 {
        stats.n_flips += 1;
        stats.n_garments_flipped += n as u64; // pieces, not garments — projection metric
        stats.flip_loss_injected += loss_injected;
    }
    debug_assert!(
        crate::grouped::invariant_holds(&sep.prob.layout, spec),
        "[ANNL] projection failed to establish the group invariant"
    );
    n
}

#[inline]
fn d2(a: (f32, f32), b: (f32, f32)) -> f32 {
    (a.0 - b.0).powi(2) + (a.1 - b.1).powi(2)
}

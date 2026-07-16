//! Exact orientation re-solve ("resolve" mode).
//!
//! At fixed piece positions, the total collision quantity as a function of the
//! per-garment orientation bits decomposes into garment-pair terms plus linear
//! terms (vs. ungrouped pieces and the container). Production markers carry few
//! garments (typically 3–10), so the whole assignment space (2^G) is enumerated
//! *exactly* and the layout is flipped to the argmin — or left untouched when
//! the current assignment is already optimal (the frequent, free case). This
//! turns the orientation move from a random walk into an optimal re-solve with
//! a built-in no-op: it cannot over-flip.
//!
//! The energy uses the same pole-based overlap proxy the separator's evaluator
//! optimizes (`quantify`-family), thresholded so only real penetrations count.
//! Optimal-at-fixed-positions is still myopic w.r.t. subsequent repacking; the
//! caller triggers it where geometry is meaningful (after successful shrinks)
//! and the infeasible-solution pool remains the final judge.

use crate::consts::OVERLAP_PROXY_EPSILON_DIAM_RATIO;
use crate::grouped::{is_dir_180, GroupedOrientationSpec};
use crate::optimizer::separator::Separator;
use crate::quantify::calc_shape_penalty;
use crate::quantify::overlap_proxy::overlap_area_proxy;
use crate::util::listener::SearchStats;
use jagua_rs::entities::{Instance, PItemKey};
use jagua_rs::geometry::geo_enums::RotationRange;
use jagua_rs::geometry::geo_traits::TransformableFrom;
use jagua_rs::geometry::primitives::{Rect, SPolygon};
use jagua_rs::geometry::DTransformation;
use log::{debug, info};

/// 2^14 = 16384 assignment evaluations — microseconds against a sparse table.
/// Markers with more physical garments than this skip the re-solve entirely.
const MAX_RESOLVE_GARMENTS: usize = 14;

/// One physical garment: its placed copies with both candidate poses.
struct GarmentCell {
    /// Placement keys, one per piece of this garment.
    pks: Vec<PItemKey>,
    /// Per piece: shape at [class-0 pose, class-180 pose].
    shapes: Vec<[SPolygon; 2]>,
    /// Per piece: transform at [class-0 pose, class-180 pose].
    dts: Vec<[DTransformation; 2]>,
    /// Current orientation class of this garment (true = 180).
    current: bool,
}

/// Re-solves the garment orientation assignment at the current positions.
/// Returns true iff at least one garment was flipped.
pub(crate) fn orientation_resolve(
    sep: &mut Separator,
    spec: &GroupedOrientationSpec,
    stats: &mut SearchStats,
) -> bool {
    stats.n_resolves += 1;

    let Some(cells) = build_cells(sep, spec) else {
        return false;
    };
    let g = cells.len();
    if g == 0 || g > MAX_RESOLVE_GARMENTS {
        return false;
    }

    // ── energy tables ────────────────────────────────────────────────────────
    // Linear terms: each garment pose vs. the container and vs. every piece not
    // belonging to any cell (direction-free items keep their current pose).
    let cbox = sep.prob.layout.container.outer_cd.bbox;
    let cell_pk_set: Vec<PItemKey> = cells.iter().flat_map(|c| c.pks.iter().copied()).collect();
    let fixed_shapes: Vec<&SPolygon> = sep
        .prob
        .layout
        .placed_items
        .iter()
        .filter(|(pk, _)| !cell_pk_set.contains(pk))
        .map(|(_, pi)| pi.shape.as_ref())
        .collect();

    let mut lin = vec![[0.0f32; 2]; g];
    for (k, cell) in cells.iter().enumerate() {
        for b in 0..2 {
            for shape in cell.shapes.iter().map(|s| &s[b]) {
                lin[k][b] += container_energy(shape, cbox);
                for fixed in &fixed_shapes {
                    lin[k][b] += pair_energy(shape, fixed);
                }
            }
        }
    }

    // Pairwise terms: garment k vs. garment l for each of the four bit combos.
    // Sparse: most garment pairs never interact in any pose (bbox prefilter).
    let mut pair: Vec<(usize, usize, [[f32; 2]; 2])> = Vec::new();
    for k in 0..g {
        for l in (k + 1)..g {
            let mut e = [[0.0f32; 2]; 2];
            let mut any = false;
            for bk in 0..2 {
                for bl in 0..2 {
                    let mut acc = 0.0;
                    for sk in cells[k].shapes.iter().map(|s| &s[bk]) {
                        for sl in cells[l].shapes.iter().map(|s| &s[bl]) {
                            acc += pair_energy(sk, sl);
                        }
                    }
                    e[bk][bl] = acc;
                    any |= acc > 0.0;
                }
            }
            if any {
                pair.push((k, l, e));
            }
        }
    }

    // ── exact enumeration, biased to no-op on ties ───────────────────────────
    let current_mask: usize = cells
        .iter()
        .enumerate()
        .map(|(k, c)| (c.current as usize) << k)
        .sum();
    let mut best_mask = current_mask;
    let mut best_e = f32::INFINITY;
    for mask in 0..(1usize << g) {
        let mut e = 0.0f32;
        for k in 0..g {
            e += lin[k][(mask >> k) & 1];
        }
        for &(k, l, ref tbl) in &pair {
            e += tbl[(mask >> k) & 1][(mask >> l) & 1];
        }
        if e < best_e - f32::EPSILON || (mask == current_mask && e <= best_e) {
            best_e = e;
            best_mask = mask;
        }
    }

    if best_mask == current_mask {
        debug!("[RSLV] current assignment already optimal (E = {best_e:.3})");
        return false; // the frequent, free outcome
    }

    // ── apply the delta ──────────────────────────────────────────────────────
    let mut n_changed = 0usize;
    let mut loss_injected = 0.0f32;
    for (k, cell) in cells.iter().enumerate() {
        let new_bit = (best_mask >> k) & 1 == 1;
        if new_bit == cell.current {
            continue;
        }
        for (i, &pk) in cell.pks.iter().enumerate() {
            let new_pk = sep.move_item(pk, cell.dts[i][new_bit as usize]);
            loss_injected += sep.ct.get_loss(new_pk);
        }
        n_changed += 1;
    }
    info!(
        "[RSLV] re-solved orientations: {n_changed}/{g} garments flipped (E {:.3} -> {:.3}), loss injected {:.3}",
        energy_of(current_mask, &lin, &pair),
        best_e,
        loss_injected,
    );
    stats.n_flips += 1;
    stats.n_garments_flipped += n_changed as u64;
    stats.flip_loss_injected += loss_injected;
    debug_assert!(
        crate::grouped::invariant_holds(&sep.prob.layout, spec),
        "[RSLV] group orientation invariant broken by re-solve"
    );
    true
}

fn energy_of(mask: usize, lin: &[[f32; 2]], pair: &[(usize, usize, [[f32; 2]; 2])]) -> f32 {
    let mut e = 0.0;
    for (k, l) in lin.iter().enumerate() {
        e += l[(mask >> k) & 1];
    }
    for &(k, l, ref tbl) in pair {
        e += tbl[(mask >> k) & 1][(mask >> l) & 1];
    }
    e
}

/// Overlap energy between two posed shapes: the search's own pole proxy,
/// bbox-prefiltered and floor-thresholded so separated pairs contribute zero.
fn pair_energy(a: &SPolygon, b: &SPolygon) -> f32 {
    if Rect::intersection(a.bbox, b.bbox).is_none() {
        return 0.0;
    }
    let epsilon = f32::max(a.diameter, b.diameter) * OVERLAP_PROXY_EPSILON_DIAM_RATIO;
    let proxy = overlap_area_proxy(a.surrogate(), b.surrogate(), epsilon);
    if proxy <= epsilon.powi(2) {
        return 0.0; // at or below the proxy's noise floor: treat as separated
    }
    proxy.sqrt() * calc_shape_penalty(a, b)
}

/// Container-exterior energy of a posed shape (zero when fully inside).
fn container_energy(s: &SPolygon, cbox: Rect) -> f32 {
    let outside = match Rect::intersection(s.bbox, cbox) {
        Some(r) => s.bbox.area() - r.area(),
        None => s.bbox.area(),
    };
    if outside <= f32::EPSILON * s.bbox.area() {
        return 0.0;
    }
    2.0 * outside.sqrt() * calc_shape_penalty(s, s)
}

/// Clusters the placed copies into physical garments, split per direction class
/// so the current assignment is well-defined (guaranteed by the group counting
/// invariant), and precomputes both candidate poses per piece.
fn build_cells(sep: &Separator, spec: &GroupedOrientationSpec) -> Option<Vec<GarmentCell>> {
    // copies per (item_id): (pk, is180, centroid, translation, rotation)
    let n_items = sep.instance.items.len();
    let mut copies: Vec<Vec<(PItemKey, bool, (f32, f32))>> = vec![Vec::new(); n_items];
    for (pk, pi) in sep.prob.layout.placed_items.iter() {
        let c = pi.shape.centroid();
        copies[pi.item_id].push((pk, is_dir_180(pi.d_transf.rotation()), (c.0, c.1)));
    }

    let mut cells: Vec<GarmentCell> = Vec::new();
    for group in &spec.groups {
        // Anchor member: smallest per_garment (largest area on ties) — its copy
        // chunks define the garment centroids within each direction class.
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

        for class180 in [false, true] {
            let mut anchors: Vec<(PItemKey, bool, (f32, f32))> = copies[anchor_id]
                .iter()
                .filter(|(_, d, _)| *d == class180)
                .copied()
                .collect();
            if anchors.is_empty() {
                continue;
            }
            debug_assert!(anchors.len() % anchor_pg == 0, "invariant: whole garments per class");
            let n_cells = anchors.len() / anchor_pg;
            anchors.sort_by(|a, b| a.2 .0.total_cmp(&b.2 .0));
            // one cell per chunk of anchor copies; centroid = chunk mean
            let mut cell_centers: Vec<(f32, f32)> = Vec::with_capacity(n_cells);
            let mut cell_pks: Vec<Vec<PItemKey>> = Vec::with_capacity(n_cells);
            for chunk in anchors.chunks(anchor_pg) {
                let cx = chunk.iter().map(|c| c.2 .0).sum::<f32>() / chunk.len() as f32;
                let cy = chunk.iter().map(|c| c.2 .1).sum::<f32>() / chunk.len() as f32;
                cell_centers.push((cx, cy));
                cell_pks.push(chunk.iter().map(|c| c.0).collect());
            }
            // assign every other member's class copies: nearest center w/ capacity
            for &(item_id, pg) in &group.members {
                if item_id == anchor_id {
                    continue;
                }
                let mut cls: Vec<(PItemKey, bool, (f32, f32))> = copies[item_id]
                    .iter()
                    .filter(|(_, d, _)| *d == class180)
                    .copied()
                    .collect();
                if cls.len() != n_cells * pg {
                    return None; // invariant violated — bail out, never guess
                }
                cls.sort_by(|a, b| a.2 .0.total_cmp(&b.2 .0));
                let mut cap = vec![pg; n_cells];
                for (pk, _, c) in cls {
                    let best = (0..n_cells)
                        .filter(|&i| cap[i] > 0)
                        .min_by(|&i, &j| {
                            d2(c, cell_centers[i]).total_cmp(&d2(c, cell_centers[j]))
                        })
                        .expect("capacity sums to copy count");
                    cap[best] -= 1;
                    cell_pks[best].push(pk);
                }
            }
            // materialize cells: both poses per piece
            for pks in cell_pks {
                let mut shapes = Vec::with_capacity(pks.len());
                let mut dts = Vec::with_capacity(pks.len());
                for &pk in &pks {
                    let pi = &sep.prob.layout.placed_items[pk];
                    let item = sep.instance.item(pi.item_id);
                    let RotationRange::Discrete(allowed) = &item.allowed_rotation else {
                        return None;
                    };
                    let cur_r = pi.d_transf.rotation();
                    let cur_is180 = is_dir_180(cur_r);
                    let Some(opp_r) = allowed.iter().copied().find(|r| is_dir_180(*r) != cur_is180)
                    else {
                        return None;
                    };
                    let c = pi.shape.centroid();
                    let t = pi.d_transf.translation();
                    let flipped =
                        DTransformation::new(opp_r, (2.0 * c.0 - t.0, 2.0 * c.1 - t.1));
                    let mut flipped_shape = item.shape_cd.as_ref().clone();
                    flipped_shape.transform_from(item.shape_cd.as_ref(), &flipped.compose());
                    let (s0, s1, d0, d1) = if cur_is180 {
                        (flipped_shape, pi.shape.as_ref().clone(), flipped, pi.d_transf)
                    } else {
                        (pi.shape.as_ref().clone(), flipped_shape, pi.d_transf, flipped)
                    };
                    shapes.push([s0, s1]);
                    dts.push([d0, d1]);
                }
                cells.push(GarmentCell {
                    pks,
                    shapes,
                    dts,
                    current: class180,
                });
            }
        }
    }
    Some(cells)
}

#[inline]
fn d2(a: (f32, f32), b: (f32, f32)) -> f32 {
    (a.0 - b.0).powi(2) + (a.1 - b.1).powi(2)
}

use crate::config::ExplorationConfig;
use crate::grouped::GroupedOrientationSpec;
use crate::optimizer::separator::{Separator, SeparatorConfig};
use crate::sample::uniform_sampler::convert_sample_to_closest_feasible;
use crate::util::listener::{ReportType, SearchStats, SolutionListener, StatsPhase};
use crate::util::terminator::Terminator;
use crate::FMT;
use float_cmp::approx_eq;
use itertools::Itertools;
use jagua_rs::collision_detection::hazards::HazardEntity;
use jagua_rs::entities::{Instance, Layout, PItemKey};
use jagua_rs::geometry::geo_enums::RotationRange;
use jagua_rs::geometry::geo_traits::CollidesWith;
use jagua_rs::geometry::DTransformation;
use jagua_rs::probs::spp::entities::{SPInstance, SPSolution};
use log::{debug, info, warn};
use ordered_float::OrderedFloat;
use rand::prelude::{Distribution, IndexedRandom, IteratorRandom};
use rand::RngExt;
use rand_distr::Normal;
use slotmap::SecondaryMap;
use std::cmp::Reverse;

/// Algorithm 12 from https://doi.org/10.48550/arXiv.2509.13329
///
/// `grouped`: optional grouped-orientation spec. When present, disruption events may
/// flip whole garment groups (see `crate::grouped`); when `None`, behavior is
/// identical to upstream.
pub fn exploration_phase(instance: &SPInstance, sep: &mut Separator, sol_listener: &mut impl SolutionListener, term: &impl Terminator, config: &ExplorationConfig, grouped: Option<&GroupedOrientationSpec>) -> Vec<SPSolution> {
    let mut current_width = sep.prob.strip_width();
    let mut best_width = current_width;

    let mut feasible_sols = vec![sep.prob.save()];

    sol_listener.report(ReportType::ExplFeas, &feasible_sols[0], instance);
    info!("[EXPL] starting optimization with initial width: {:.3} ({:.3}%)",current_width,sep.prob.density() * 100.0);

    let mut infeas_sol_pool: Vec<(SPSolution, f32)> = vec![];

    debug_assert!(
        grouped.map_or(true, |s| crate::grouped::invariant_holds(&sep.prob.layout, s)),
        "[EXPL] initial solution violates the group orientation invariant"
    );

    // Phase statistics: plain integers on this (master) thread; reported once at phase end.
    let mut stats = SearchStats { grouped_active: grouped.is_some(), ..SearchStats::default() };
    let phase_start = jagua_rs::Instant::now();
    // What the previous disruption was, to attribute the following separation outcome.
    let mut last_disruption: Option<DisruptionKind> = None;

    while !term.kill() {
        // Attempt to separate the current layout
        let local_best = sep.separate(term, sol_listener);
        let total_loss = local_best.1.get_total_loss();

        // Attribute this separation's outcome to the disruption that preceded it.
        if let Some(kind) = last_disruption.take() {
            if total_loss == 0.0 {
                match kind {
                    DisruptionKind::Swap => stats.post_swap_sep_success += 1,
                    DisruptionKind::Flip => stats.post_flip_sep_success += 1,
                }
            }
        }

        if total_loss == 0.0 {
            // If successfully separated
            if current_width < best_width {
                info!("[EXPL] feasible solution found! (width: {:.3}, dens: {:.3}%)",current_width,sep.prob.density() * 100.0);
                best_width = current_width;
                feasible_sols.push(local_best.0.clone());
                sol_listener.report(ReportType::ExplFeas, &local_best.0, instance);
            }
            // Shrink the strip width and clear the infeasible solution pool
            let next_width = current_width * (1.0 - config.shrink_step);
            info!("[EXPL] shrinking strip by {}%: {:.3} -> {:.3}", config.shrink_step * 100.0, current_width, next_width);
            sep.change_strip_width(next_width, None);
            current_width = next_width;
            infeas_sol_pool.clear();
            stats.n_shrinks += 1;
            // Proactive flip: partition moves are made while the layout is plastic
            // (right after a proven-feasible width, where repair is cheapest) —
            // not only at stalls, where the layout is at its tightest.
            if let Some(spec) = grouped {
                let in_window = phase_start.elapsed().as_secs_f32()
                    < spec.flip.window * config.time_limit.as_secs_f32();
                if in_window && sep.rng.random::<f32>() < spec.flip.p_flip
                    && disrupt_by_group_flip(sep, spec, &mut stats)
                {
                    stats.n_disruptions += 1;
                    last_disruption = Some(DisruptionKind::Flip);
                }
            }
        } else {
            info!("[EXPL] unable to reach feasibility (width: {:.3}, dens: {:.3}%, min loss: {:.3})", current_width, sep.prob.density() * 100.0, FMT().fmt2(total_loss));
            sol_listener.report(ReportType::ExplInfeas, &local_best.0, instance);

            // Separation was not successful add it to the pool of infeasible solutions
            match infeas_sol_pool.binary_search_by(|(_, o)| o.partial_cmp(&total_loss).unwrap()) {
                Ok(idx) | Err(idx) => infeas_sol_pool.insert(idx, (local_best.0.clone(), total_loss)),
            }

            if infeas_sol_pool.len() >= config.max_conseq_failed_attempts.unwrap_or(usize::MAX) {
                info!("[EXPL] max consecutive failed attempts ({}), terminating", infeas_sol_pool.len());
                break;
            }

            // Restore to a random solution from the pool, with better solutions having more chance to be selected
            let selected_sol = {
                // Sample a value in range [0.0, 1.0[ from a normal distribution
                let distribution = Normal::new(0.0, config.solution_pool_distribution_stddev).unwrap();
                let sample = distribution.sample(&mut sep.rng).abs().min(0.999);
                // Map it to an index in the infeasible solution pool (better solutions are at the start of the pool)
                let selected_idx = (sample * infeas_sol_pool.len() as f32) as usize;

                let (selected_sol, loss) = &infeas_sol_pool[selected_idx];
                info!("[EXPL] starting solution {}/{} selected from solution pool (l: {}) to disrupt", selected_idx, infeas_sol_pool.len(), FMT().fmt2(*loss));
                selected_sol
            };

            // Rollback to this solution and disrupt it: a group flip (grouped mode,
            // with probability p_flip) or the swap-two-large-items move.
            sep.rollback(selected_sol, None);
            stats.n_disruptions += 1;
            let flipped = match grouped {
                Some(spec)
                    if phase_start.elapsed().as_secs_f32()
                        < spec.flip.window * config.time_limit.as_secs_f32()
                        && sep.rng.random::<f32>() < spec.flip.p_flip =>
                {
                    disrupt_by_group_flip(sep, spec, &mut stats)
                }
                _ => false,
            };
            if flipped {
                last_disruption = Some(DisruptionKind::Flip);
            } else {
                disrupt_solution(sep, config);
                stats.n_swaps += 1;
                last_disruption = Some(DisruptionKind::Swap);
            }
        }
    }

    info!("[EXPL] finished, best feasible solution: width: {:.3} ({:.3}%)",best_width,feasible_sols.last().unwrap().density(instance) * 100.0);

    // Assemble and emit phase statistics (2nd-to-none hot-path cost: this runs once).
    stats.n_separate_calls = sep.cum_separate_calls;
    stats.total_moves = sep.cum_moves;
    stats.total_evals = sep.cum_evals;
    stats.phase_secs = phase_start.elapsed().as_secs_f32();
    sol_listener.on_search_stats(StatsPhase::Exploration, &stats);

    feasible_sols
}

/// Which kind of disruption was applied (for outcome attribution in `SearchStats`).
#[derive(Debug, Clone, Copy)]
enum DisruptionKind {
    Swap,
    Flip,
}

/// Grouped-orientation disruption: flips `n_garments_per_flip` whole garments
/// 180° in place (see `crate::grouped`). Returns whether at least one garment was
/// flipped; on false the caller falls back to the swap disruption.
fn disrupt_by_group_flip(
    sep: &mut Separator,
    spec: &crate::grouped::GroupedOrientationSpec,
    stats: &mut SearchStats,
) -> bool {
    let mut any = false;
    for _ in 0..spec.flip.n_garments_per_flip {
        any |= flip_one_garment(sep, spec, stats);
    }
    any
}

/// Flips one garment: picks a (group, source direction) uniformly among the valid
/// pairs, then — exploiting piece interchangeability — assembles the cheapest
/// "garment" to repair: an anchor copy of the group's largest piece plus, per
/// member piece type, its `per_garment` copies nearest the anchor (spatial
/// coherence keeps the injected overlap local). Each selected copy is rotated to
/// the opposite allowed angle about its own placed centroid: with world centroid
/// C and translation t, the in-place pose is r' = opposite(r), t' = 2C - t
/// (since R_{r+pi}(c) = -R_r(c)). The following separation repairs the overlap;
/// rotation-locked moves guarantee the flip cannot be undone piecemeal, and the
/// infeasible-solution pool provides the accept/reject pressure.
fn flip_one_garment(
    sep: &mut Separator,
    spec: &crate::grouped::GroupedOrientationSpec,
    stats: &mut SearchStats,
) -> bool {
    use crate::grouped::is_dir_180;

    // Census of the group members' placed copies: item_id -> (pk, is180, centroid).
    let n_items = sep.instance.items.len();
    let mut copies: Vec<Vec<(PItemKey, bool, (f32, f32))>> = vec![Vec::new(); n_items];
    for (pk, pi) in sep.prob.layout.placed_items.iter() {
        let c = pi.shape.centroid();
        copies[pi.item_id].push((pk, is_dir_180(pi.d_transf.rotation()), (c.0, c.1)));
    }

    // Valid (group, source-direction) pairs: at least one whole garment faces it.
    let mut cands: Vec<(usize, bool)> = Vec::new();
    for (gi, g) in spec.groups.iter().enumerate() {
        let (item0, pg0) = g.members[0];
        let n0 = copies[item0].iter().filter(|(_, d, _)| !d).count() / pg0;
        if n0 >= 1 {
            cands.push((gi, false));
        }
        if g.quantity.saturating_sub(n0) >= 1 {
            cands.push((gi, true));
        }
    }
    let Some(&(gi, src180)) = cands.choose(&mut sep.rng) else {
        return false;
    };
    let group = &spec.groups[gi];

    // Anchor: a random source-direction copy of the group's largest-area member.
    let largest = group
        .members
        .iter()
        .max_by(|a, b| {
            sep.instance
                .item(a.0)
                .shape_cd
                .area
                .total_cmp(&sep.instance.item(b.0).shape_cd.area)
        })
        .expect("groups have >= 1 member (validated)")
        .0;
    let Some(&(_, _, anchor)) = copies[largest]
        .iter()
        .filter(|(_, d, _)| *d == src180)
        .choose(&mut sep.rng)
    else {
        return false;
    };

    // Select per_garment nearest source-direction copies of every member and
    // compute their flipped poses. No mutation until the whole garment resolves.
    let dist2 = |a: (f32, f32)| (a.0 - anchor.0).powi(2) + (a.1 - anchor.1).powi(2);
    let mut moves: Vec<(PItemKey, DTransformation)> = Vec::new();
    for &(item_id, pg) in &group.members {
        let RotationRange::Discrete(allowed) = &sep.instance.item(item_id).allowed_rotation
        else {
            return false; // group members must carry discrete two-class rotations
        };
        let Some(target) = allowed.iter().copied().find(|r| is_dir_180(*r) != src180) else {
            return false;
        };
        let mut same: Vec<(PItemKey, (f32, f32))> = copies[item_id]
            .iter()
            .filter(|(_, d, _)| *d == src180)
            .map(|&(pk, _, c)| (pk, c))
            .collect();
        if same.len() < pg {
            debug_assert!(false, "[FLIP] fewer source copies than per_garment — invariant broken");
            return false;
        }
        same.sort_by(|a, b| dist2(a.1).total_cmp(&dist2(b.1)));
        for &(pk, c) in same.iter().take(pg) {
            let t = sep.prob.layout.placed_items[pk].d_transf.translation();
            moves.push((
                pk,
                DTransformation::new(target, (2.0 * c.0 - t.0, 2.0 * c.1 - t.1)),
            ));
        }
    }

    let n_pieces = moves.len();
    let mut loss_injected = 0.0;
    for (pk, dt) in moves {
        let new_pk = sep.move_item(pk, dt);
        loss_injected += sep.ct.get_loss(new_pk);
    }
    info!(
        "[FLIP] flipped garment of group {gi} ({n_pieces} pieces, {} -> {}), loss injected: {:.3}",
        if src180 { "180" } else { "0" },
        if src180 { "0" } else { "180" },
        loss_injected,
    );
    stats.n_flips += 1;
    stats.n_garments_flipped += 1;
    stats.flip_loss_injected += loss_injected;
    debug_assert!(
        crate::grouped::invariant_holds(&sep.prob.layout, spec),
        "[FLIP] group orientation invariant broken by flip"
    );
    true
}

fn disrupt_solution(sep: &mut Separator, config: &ExplorationConfig) {
    if sep.prob.layout.placed_items.len() < 2 {
        warn!("[DSRP] cannot disrupt solution with less than 2 items");
        return;
    }

    // The general idea is to disrupt a solution by swapping two 'large' items in the layout.
    // 'Large' items are those whose convex hull area falls within a certain top percentile
    // of the total convex hull area of all items in the layout.

    // Step 1: Define what constitutes a 'large' item.

    // Calculate the total convex hull area of all items, considering quantities.
    let total_convex_hull_area: f32 = sep
        .prob
        .instance
        .items
        .iter()
        .map(|(item, quantity)| item.shape_cd.surrogate().convex_hull_area * (*quantity as f32))
        .sum();

    let cutoff_threshold_area = total_convex_hull_area * config.large_item_ch_area_cutoff_percentile;

    // Sort items by convex hull area in descending order.
    let sorted_items_by_ch_area = sep
        .prob
        .instance
        .items
        .iter()
        .sorted_by_key(|(item, _)| Reverse(OrderedFloat(item.shape_cd.surrogate().convex_hull_area)))
        .peekable();

    let mut cumulative_ch_area = 0.0;
    let mut ch_area_cutoff = 0.0;

    // Iterate through items, accumulating their convex hull areas until the cumulative sum
    // exceeds the cutoff_threshold_area. The convex hull area of the item that causes
    // this excess becomes the ch_area_cutoff.
    for (item, quantity) in sorted_items_by_ch_area {
        let item_ch_area = item.shape_cd.surrogate().convex_hull_area;
        cumulative_ch_area += item_ch_area * (*quantity as f32);
        if cumulative_ch_area > cutoff_threshold_area {
            ch_area_cutoff = item_ch_area;
            debug!("[DSRP] cutoff ch area: {}, for item id: {}, bbox: {:?}",ch_area_cutoff, item.id, item.shape_cd.bbox);
            break;
        }
    }

    // Step 2: Select two 'large' items and 'swap' them.

    let large_items = sep.prob.layout.placed_items.iter()
        .filter(|(_, pi)| pi.shape.surrogate().convex_hull_area >= ch_area_cutoff);

    //Choose a first item with a large enough convex hull
    let (pk1, pi1) = large_items.clone().choose(&mut sep.rng).expect("[DSRP] failed to choose first item");

    //Choose a second item with a large enough convex hull and different enough from the first.
    //If no such item is found, choose a random one.
    let (pk2, pi2) = large_items.clone()
        .filter(|(_, pi)|
            // Ensure the second item is different from the first
            !approx_eq!(f32, pi.shape.area,pi1.shape.area, epsilon = pi1.shape.area * 0.01) &&
                !approx_eq!(f32, pi.shape.diameter, pi1.shape.diameter, epsilon = pi1.shape.diameter * 0.01)
        )
        .choose(&mut sep.rng)
        .or_else(|| {
            sep.prob.layout.placed_items.iter()
                .filter(|(pk, _)| *pk != pk1) // Ensure the second item is not the same as the first
                .choose(&mut sep.rng)
        }) // As a fallback, choose any item
        .expect("[EXPL] failed to choose second item for disruption");

    // Step 3: Swap the two items' positions in the layout.

    let dt1_old = pi1.d_transf;
    let dt2_old = pi2.d_transf;

    // Make sure the swaps do not violate feasibility (rotation).
    let dt1_new = convert_sample_to_closest_feasible(dt2_old, sep.prob.instance.item(pi1.item_id));
    let dt2_new = convert_sample_to_closest_feasible(dt1_old, sep.prob.instance.item(pi2.item_id));

    info!("[EXPL] disrupting by swapping two large items (id: {} <-> {})", pi1.item_id, pi2.item_id);

    let pk1 = sep.move_item(pk1, dt1_new);
    let pk2 = sep.move_item(pk2, dt2_new);


    // Step 4: Move all items that are practically contained by one of the swapped items to the "empty space" created by the moved item.
    //         This is particularly important when huge items are swapped with smaller items. 
    //         The huge item will create a large empty space and many of the items which previously 
    //         surrounded the smaller one will be contained by the huge one.
    {
        // transformation to convert the contained items' position (relative to the old and new positions of the swapped items)
        let converting_transformation = dt1_new.compose().inverse()
            .transform(&dt1_old.compose());

        for c1_pk in practically_contained_items(&sep.prob.layout, pk1).into_iter().filter(|c1_pk| *c1_pk != pk2) {
            let c1_pi = &sep.prob.layout.placed_items[c1_pk];

            let new_dt = c1_pi.d_transf
                .compose()
                .transform(&converting_transformation)
                .decompose();

            //Ensure the sure the new position is feasible
            let new_feasible_dt = convert_sample_to_closest_feasible(new_dt, sep.prob.instance.item(c1_pi.item_id));
            sep.move_item(c1_pk, new_feasible_dt);
        }
    }

    // Do the same for the second item, but using the second transformation
    {
        let converting_transformation = dt2_new.compose().inverse()
            .transform(&dt2_old.compose());

        for c2_pk in practically_contained_items(&sep.prob.layout, pk2).into_iter().filter(|c2_pk| *c2_pk != pk1) {
            let c2_pi = &sep.prob.layout.placed_items[c2_pk];
            let new_dt = c2_pi.d_transf
                .compose()
                .transform(&converting_transformation)
                .decompose();

            //make sure the new position is feasible
            let new_feasible_dt = convert_sample_to_closest_feasible(new_dt, sep.prob.instance.item(c2_pi.item_id));
            sep.move_item(c2_pk, new_feasible_dt);
        }
    }
}

/// Collects all items which point of inaccessibility (POI) is contained by pk_c's shape.
fn practically_contained_items(layout: &Layout, pk_c: PItemKey) -> Vec<PItemKey> {
    let pi_c = &layout.placed_items[pk_c];
    // Detect all collisions with the item pk_c's shape.
    let mut collector = SecondaryMap::new();
    layout.cde().collect_poly_collisions(&pi_c.shape, &mut collector);

    // Filter out the items that have their POI contained by pk_c's shape.
    collector.iter()
        .filter_map(|(_,he)| {
            match he {
                HazardEntity::PlacedItem { pk, .. } => Some(*pk),
                _ => None
            }
        })
        .filter(|pk| *pk != pk_c) // Ensure we don't include the item itself
        .filter(|pk| {
            // Check if the POI of the item is contained by pk_c's shape
            let poi = layout.placed_items[*pk].shape.poi;
            pi_c.shape.collides_with(&poi.center)
        })
        .collect_vec()
}
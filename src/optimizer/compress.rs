use crate::config::{CompressionConfig, ShrinkDecayStrategy};
use crate::optimizer::separator::Separator;
use crate::optimizer::small_relocate::{classify_colliders, relocate_colliding_smalls, tidy_smalls};
use crate::util::listener::{ReportType, SearchStats, SolutionListener, StatsPhase};
use crate::util::terminator::Terminator;
use jagua_rs::probs::spp::entities::{SPInstance, SPSolution};
use jagua_rs::Instant;
use log::info;
use rand::{Rng, RngExt};

/// Algorithm 13 from https://doi.org/10.48550/arXiv.2509.13329
pub fn compression_phase(
    instance: &SPInstance,
    sep: &mut Separator,
    init_sol: &SPSolution,
    sol_listener: &mut impl SolutionListener,
    term: &impl Terminator,
    config: &CompressionConfig
) -> SPSolution {
    if config.window_mode {
        info!("[CMPR] window-ladder mode: compression skipped (length pressure would drain the window)");
        return init_sol.clone();
    }
    let mut best_sol = init_sol.clone();
    let start = Instant::now();
    let mut n_failed_attempts = 0;
    let mut n_successes: u64 = 0;
    // Phase statistics: plain integers on this (master) thread; reported once at phase end.
    let mut stats = SearchStats { grouped_active: sep.locked_items.is_some(), ..SearchStats::default() };

    // Create the function to calculate the shrink step size.
    let shrink_step_size = |n_failed_attempts: i32| -> f32 {
        match config.shrink_decay {
            ShrinkDecayStrategy::TimeBased => {
                let range = config.shrink_range.1 - config.shrink_range.0;
                let elapsed = start.elapsed();
                let ratio = elapsed.as_secs_f32() / config.time_limit.as_secs_f32();
                config.shrink_range.0 + ratio * range
            }
            ShrinkDecayStrategy::FailureBased(r) => {
                config.shrink_range.0 * r.powi(n_failed_attempts)
            }
        }
    };

    // As long as the shrink step size is above the minimum, keep attempting to compress
    while !term.kill() && let step = shrink_step_size(n_failed_attempts) && step >= config.shrink_range.1 {
        match attempt_to_compress(sep, &best_sol, step, config.prop_shrink, term, sol_listener, &mut stats) {
            Some(compacted_sol) => {
                info!("[CMPR] success at {:.3}% ({:.3} | {:.3}%)", step * 100.0, compacted_sol.strip_width(), compacted_sol.density(instance) * 100.0);
                sol_listener.report(ReportType::CmprFeas, &compacted_sol, instance);
                best_sol = compacted_sol;
                n_successes += 1;
            }
            None => {
                info!("[CMPR] failed at {:.3}%", step * 100.0);
                n_failed_attempts += 1;
            }
        }
    }
    info!("[CMPR] finished, compressed from {:.3}% to {:.3}% (+{:.3}%)", init_sol.density(instance) * 100.0, best_sol.density(instance) * 100.0, (best_sol.density(instance) - init_sol.density(instance)) * 100.0);

    // Assemble and emit phase statistics (runs once, at phase end).
    stats.n_separate_calls = sep.cum_separate_calls;
    stats.total_moves = sep.cum_moves;
    stats.total_evals = sep.cum_evals;
    stats.voluntary_moves = sep.cum_voluntary_moves;
    stats.n_yield_volleys = sep.cum_yield_volleys;
    stats.n_panels_yielded = sep.cum_panels_yielded;
    stats.n_sep_with_yield = sep.cum_sep_with_yield;
    stats.n_sep_success_with_yield = sep.cum_sep_success_with_yield;
    stats.n_shrinks = n_successes;
    stats.n_small_items = sep.prob.layout.placed_items.iter()
        .filter(|(_, pi)| sep.small_items[pi.item_id]).count() as u64;
    stats.phase_secs = start.elapsed().as_secs_f32();
    sol_listener.on_search_stats(StatsPhase::Compression, &stats);

    best_sol
}


fn attempt_to_compress(sep: &mut Separator, init_sol: &SPSolution, r_shrink: f32, prop_shrink: bool, term: &impl Terminator, sol_listener: &mut impl SolutionListener, stats: &mut SearchStats) -> Option<SPSolution> {
    // Restore to the initial solution and width
    sep.change_strip_width(init_sol.strip_width(), None);
    sep.rollback(init_sol, None);

    // Shrink the container by the provided amount at a random position
    let new_width = init_sol.strip_width() * (1.0 - r_shrink);
    if prop_shrink {
        sep.change_strip_width_prop(new_width);
    } else {
        let split_pos = {
            let lo = if sep.freeze_clamp > 0.0 { (sep.freeze_clamp + 1.0).min(sep.prob.strip_width() * 0.99) } else { 0.0 };
            sep.rng.random_range(lo..sep.prob.strip_width())
        };
        sep.change_strip_width(new_width, Some(split_pos));
    }

    // Small-piece mechanism T (see `small_relocate`): tidy before separating.
    if sep.config.small.tidy_mode > 0 {
        tidy_smalls(sep, stats, sep.config.small.tidy_mode >= 2);
        if sep.ct.get_total_loss() == 0.0 {
            return Some(sep.prob.save());
        }
    }

    // Try to separate layout, if all collisions are eliminated, return the solution
    let (compacted_sol, ot) = sep.separate(term, sol_listener);
    if ot.get_total_loss() == 0.0 {
        return Some(compacted_sol);
    }

    // Small-piece mechanisms (see `crate::small`): classify the failure; under D, a
    // small-only colliding set gets its pieces re-placed before the attempt is written off.
    let census = classify_colliders(sep);
    census.record(stats);
    if sep.config.small.relocate_enabled() {
        sep.rollback(&compacted_sol, Some(&ot));
        if relocate_colliding_smalls(sep, census, stats).is_some_and(|n| n > 0)
            && sep.ct.get_total_loss() == 0.0
        {
            return Some(sep.prob.save());
        }
    }
    None
}

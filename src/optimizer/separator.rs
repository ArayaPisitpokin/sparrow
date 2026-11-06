use crate::optimizer::worker::{SepStats, SeparatorWorker};
use crate::optimizer::Terminator;
use crate::quantify::tracker::{CTSnapshot, CollisionTracker};
use crate::sample::search::SampleConfig;
use crate::util::assertions::tracker_matches_layout;
use crate::util::listener::{ReportType, SolutionListener};
use crate::FMT;
use itertools::Itertools;
use jagua_rs::entities::PItemKey;
use jagua_rs::geometry::DTransformation;
use jagua_rs::probs::spp::entities::{SPInstance, SPPlacement, SPProblem, SPSolution};
use jagua_rs::Instant;
use log::{debug, log, Level};
use ordered_float::OrderedFloat;
use rand::{Rng, RngExt, SeedableRng};
use rand::rngs::Xoshiro256PlusPlus;
use rayon::iter::IntoParallelRefMutIterator;
use rayon::iter::ParallelIterator;
use rayon::ThreadPool;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use crate::grouped::GroupedOrientationSpec;
use crate::optimizer::anneal::{build_anneal_map, AnnealMap};
use crate::small::{small_item_mask, SmallPieceConfig};

#[derive(Debug, Clone, Copy)]
pub struct SeparatorConfig {
    pub iter_no_imprv_limit: usize,
    pub strike_limit: usize,
    pub n_workers: usize,
    pub log_level: Level,
    pub sample_config: SampleConfig,
    /// Small-piece mechanisms (see `crate::small`); `SmallPieceConfig::OFF` = upstream.
    pub small: SmallPieceConfig,
    /// Positional freeze (2026-09-01): pieces of the warm-start layout whose bbox
    /// begins left of this line are immovable for the whole solve — never moved,
    /// never swapped, never slid by shrinks. `None` = upstream.
    pub freeze_x: Option<f32>,
}

pub struct Separator {
    pub instance: SPInstance,
    pub rng: Xoshiro256PlusPlus,
    pub prob: SPProblem,
    pub ct: CollisionTracker,
    pub workers: Vec<SeparatorWorker>,
    pub config: SeparatorConfig,
    pub thread_pool: Option<ThreadPool>,
    /// Grouped-orientation spec (drives locks, flips, resolve and anneal).
    pub grouped: Option<Arc<GroupedOrientationSpec>>,
    /// Grouped-orientation mode: per-item rotation locks (item_id -> locked).
    /// `None` = upstream behavior. Group members are locked (regular moves keep the
    /// copy's rotation); ungrouped items (e.g. direction-free pieces) stay free.
    pub locked_items: Option<Arc<[bool]>>,
    /// Whether the locks currently apply (false during an anneal window; the
    /// exploration loop stores true at projection). One relaxed load per move.
    pub locks_active: Arc<AtomicBool>,
    /// Current anneal penalty ceiling λ(t)·unit-scale, set by the exploration
    /// loop between separations; 0 disables the anneal map entirely.
    pub anneal_lambda: f32,
    /// Cumulative counters over this separator's lifetime (one optimization phase).
    /// Plain integers, only touched once per `separate()` call on the master thread.
    pub cum_separate_calls: u64,
    pub cum_moves: u64,
    pub cum_evals: u64,
    pub cum_voluntary_moves: u64,
    pub cum_yield_volleys: u64,
    pub cum_panels_yielded: u64,
    pub cum_sep_with_yield: u64,
    pub cum_sep_success_with_yield: u64,
    /// item_id -> is "small" (below the large-item cutoff, `crate::small`). Always
    /// populated so failed separations can be classified even with all mechanisms off.
    pub small_items: Arc<[bool]>,
    /// Small-piece mechanism L: lane centres for large items (empty = off).
    pub lane_centers: Arc<[f32]>,
    /// Positional freeze: registry of immovable copies, keyed by (item_id,
    /// quantized transform). Position-keyed so it survives clones and rollbacks
    /// (frozen pieces never move, so their transforms stay registry-exact).
    pub frozen: Option<Arc<std::collections::HashSet<(usize, i64, i64, i64)>>>,
    /// Max bbox x of any frozen piece (overhangs included): shrink cuts must
    /// land right of this so no slide ever displaces a frozen piece. 0 = off.
    pub freeze_clamp: f32,
}

/// Quantized identity of a placed copy (0.001 deg / 0.01 mm grid).
pub fn quant_id(item_id: usize, dt: &DTransformation) -> (usize, i64, i64, i64) {
    let t = dt.translation();
    (item_id,
     (dt.rotation() as f64 * 1e3).round() as i64,
     (t.0 as f64 * 1e2).round() as i64,
     (t.1 as f64 * 1e2).round() as i64)
}

/// Is this placed item in the frozen registry?
pub fn pi_frozen(
    frozen: &Option<Arc<std::collections::HashSet<(usize, i64, i64, i64)>>>,
    pi: &jagua_rs::entities::PlacedItem,
) -> bool {
    frozen.as_ref().is_some_and(|f| f.contains(&quant_id(pi.item_id, &pi.d_transf)))
}

impl Separator {
    pub fn new(instance: SPInstance, prob: SPProblem, mut rng: Xoshiro256PlusPlus, config: SeparatorConfig, grouped: Option<Arc<GroupedOrientationSpec>>) -> Self {
        let ct = CollisionTracker::new(&prob.layout);
        let small_items = small_item_mask(&instance, config.small.cutoff_percentile);
        debug!("[SEP] small-piece mask: {}/{} item types small (cutoff percentile {})",
            small_items.iter().filter(|s| **s).count(), small_items.len(), config.small.cutoff_percentile);
        let lane_centers: Arc<[f32]> = if config.small.lane_weight > 0.0 {
            let h = prob.layout.container.outer_cd.bbox.height();
            let mut big_heights: Vec<f32> = instance.items.iter().enumerate()
                .filter(|(id, _)| !small_items[*id])
                .map(|(_, (item, _))| item.shape_cd.bbox.height())
                .collect();
            big_heights.sort_by(f32::total_cmp);
            let median = big_heights.get(big_heights.len() / 2).copied().unwrap_or(h);
            let centers = crate::small::lane_centers(config.small.lane_count, h, median);
            log!(config.log_level, "[SEP] lane discipline: {} lanes at {:?} (median big height {:.1})", centers.len(), centers, median);
            centers.into()
        } else {
            Vec::new().into()
        };
        let locked_items: Option<Arc<[bool]>> = grouped.as_ref().map(|spec| {
            let mut locked = vec![false; instance.items.len()];
            for g in &spec.groups {
                for &(item_id, _) in &g.members {
                    locked[item_id] = true;
                }
            }
            locked.into()
        });
        // Positional freeze: register immovable copies from the (warm-start)
        // layout — everything whose bbox starts left of freeze_x, overhang included.
        let (frozen, freeze_clamp) = match config.freeze_x {
            Some(fx) => {
                let mut set = std::collections::HashSet::new();
                let mut clamp = fx;
                for (_, pi) in prob.layout.placed_items.iter() {
                    if pi.shape.bbox.x_min < fx {
                        set.insert(quant_id(pi.item_id, &pi.d_transf));
                        clamp = clamp.max(pi.shape.bbox.x_max);
                    }
                }
                log!(config.log_level, "[SEP] freeze: {} pieces immovable left of {:.1} (cut clamp {:.1})", set.len(), fx, clamp);
                (Some(Arc::new(set)), clamp)
            }
            None => (None, 0.0f32),
        };
        // Locks are live from the start except in anneal mode, where they
        // activate at the end-of-window projection.
        let locks_active = Arc::new(AtomicBool::new(
            grouped.as_ref().is_none_or(|s| !s.flip.anneal),
        ));
        let workers = (0..config.n_workers).map(|_|
            SeparatorWorker {
                instance: instance.clone(),
                prob: prob.clone(),
                ct: ct.clone(),
                rng: Xoshiro256PlusPlus::seed_from_u64(rng.random()),
                sample_config: config.sample_config,
                locked_items: locked_items.clone(),
                locks_active: locks_active.clone(),
                anneal: None,
                small: config.small,
                small_items: small_items.clone(),
                lane_centers: lane_centers.clone(),
                frozen: frozen.clone(),
            }).collect();

        let pool = if cfg!(target_arch = "wasm32") {
            // On wasm32, only the global thread pool is available
            None
        } else {
            // Create a local thread pool to keep using the same threads for the same optimization (helps the OS scheduler)
            Some(rayon::ThreadPoolBuilder::new().num_threads(config.n_workers).build().unwrap())
        };

        Self {
            prob,
            instance,
            rng,
            ct,
            workers,
            config,
            thread_pool: pool,
            grouped,
            locked_items,
            locks_active,
            anneal_lambda: 0.0,
            cum_separate_calls: 0,
            cum_moves: 0,
            cum_evals: 0,
            cum_voluntary_moves: 0,
            cum_yield_volleys: 0,
            cum_panels_yielded: 0,
            cum_sep_with_yield: 0,
            cum_sep_success_with_yield: 0,
            small_items,
            lane_centers,
            frozen,
            freeze_clamp,
        }
    }

    /// Algorithm 9 from https://doi.org/10.48550/arXiv.2509.13329
    pub fn separate(&mut self, term: &impl Terminator, sol_listener: &mut impl SolutionListener) -> (SPSolution, CTSnapshot) {
        let mut min_loss_sol = (self.prob.save(), self.ct.save());
        let mut min_loss = self.ct.get_total_loss();
        log!(self.config.log_level,"[SEP] separating at width: {:.3} and loss: {} ", self.prob.strip_width(), FMT().fmt2(min_loss));

        let mut n_strikes = 0;
        let mut n_iter = 0;
        let mut sep_stats = SepStats { total_moves: 0, total_evals: 0, voluntary_moves: 0 };
        let mut yield_volleys_this_sep: usize = 0;
        let mut separated_with_yield = false;
        let start = Instant::now();

        // As long as the strike limit is not reached, and the solution is not yet separated.
        'outer: while n_strikes < self.config.strike_limit && !term.kill() {
            let mut n_iter_no_improvement = 0;

            let initial_strike_loss = self.ct.get_total_loss();
            debug!("[SEP] [s:{n_strikes},i:{n_iter}]     init_l: {}",FMT().fmt2(initial_strike_loss));

            while n_iter_no_improvement < self.config.iter_no_imprv_limit {
                // Y (T1): on an inner-loop stall while only small pieces collide (a
                // jam), step the clear bordering panels outward to clear positions.
                // Loss-neutral on the master state; kept by the min-loss snapshot
                // only if a subsequent small-escape lowers the loss.
                if self.config.small.yield_mode > 0
                    && n_iter_no_improvement > 0
                    && n_iter_no_improvement % self.config.small.yield_stall == 0
                    && yield_volleys_this_sep < self.config.small.yield_max
                    && {
                        let c = crate::optimizer::small_relocate::classify_colliders(self);
                        c.n_small >= 2 && c.n_big <= self.config.small.yield_gate_bigs as usize
                    }
                {
                    let n = crate::optimizer::small_relocate::yield_panels_in_sep(self);
                    if n > 0 {
                        yield_volleys_this_sep += 1;
                        self.cum_yield_volleys += 1;
                        self.cum_panels_yielded += n as u64;
                    }
                }
                let (loss_before, w_loss_before) = (self.ct.get_total_loss(), self.ct.get_total_weighted_loss(),);
                sep_stats += self.move_items_multi();
                let (loss, w_loss) = (self.ct.get_total_loss(), self.ct.get_total_weighted_loss(),);

                debug!("[SEP] [s:{n_strikes},i:{n_iter}] ( ) l: {} -> {}, wl: {} -> {}, (min l: {})", FMT().fmt2(loss_before), FMT().fmt2(loss), FMT().fmt2(w_loss_before), FMT().fmt2(w_loss), FMT().fmt2(min_loss));
                debug_assert!(w_loss <= w_loss_before * 1.001 || self.anneal_lambda > 0.0, "weighted loss should not increase: {} -> {}", FMT().fmt2(w_loss), FMT().fmt2(w_loss_before));

                if loss == 0.0 {
                    //All collisions are resolved
                    log!(self.config.log_level,"[SEP] [s:{n_strikes},i:{n_iter}] (S)  min_l: {}",FMT().fmt2(loss));
                    min_loss_sol = (self.prob.save(), self.ct.save());
                    separated_with_yield = yield_volleys_this_sep > 0;
                    break 'outer;
                } else if loss < min_loss {
                    //Not all collisions are resolved, but we found a new 'best' solution
                    log!(self.config.log_level,"[SEP] [s:{n_strikes},i:{n_iter}] (*) min_l: {}",FMT().fmt2(loss));
                    sol_listener.report(ReportType::ExplImproving, &self.prob.save(), &self.instance);
                    if loss < min_loss * 0.98 {
                        //Reset the `iter_no_improvement` counter if the best solution is a substantial improvement
                        n_iter_no_improvement = 0;
                    }
                    min_loss_sol = (self.prob.save(), self.ct.save());
                    min_loss = loss;
                } else {
                    // No improvement this iteration
                    n_iter_no_improvement += 1;
                }

                // Update the GLS weights
                self.ct.update_weights();
                n_iter += 1;
            }

            if initial_strike_loss * 0.98 <= min_loss {
                // No substantial improvement during this attempt, add a strike
                n_strikes += 1;
            } else {
                // Substantial improvement, reset strike counter
                n_strikes = 0;
            }
            self.rollback(&min_loss_sol.0, Some(&min_loss_sol.1));
        }
        let secs = start.elapsed().as_secs_f32();
        // Cumulative phase stats (master thread; once per separate() call).
        self.cum_separate_calls += 1;
        self.cum_moves += sep_stats.total_moves as u64;
        self.cum_evals += sep_stats.total_evals as u64;
        self.cum_voluntary_moves += sep_stats.voluntary_moves as u64;
        if yield_volleys_this_sep > 0 {
            self.cum_sep_with_yield += 1;
            if separated_with_yield {
                self.cum_sep_success_with_yield += 1;
            }
        }
        log!(self.config.log_level, "[SEP] finished, evals/s: {} K, evals/move: {}, moves/s: {}, iter/s: {}, #workers: {}, total {:.3}s",
            (sep_stats.total_evals as f32/ (1000.0 * secs)) as usize,
            FMT().fmt2(sep_stats.total_evals as f32 / sep_stats.total_moves as f32),
            FMT().fmt2(sep_stats.total_moves as f32 / secs),
            FMT().fmt2(n_iter as f32 / secs),
            self.workers.len(),
            FMT().fmt2(secs),
        );

        // Return the best solution found: a feasible one if separation was successful, otherwise the 'least' infeasible one
        (min_loss_sol.0, min_loss_sol.1)
    }

    /// Algorithm 10 from https://doi.org/10.48550/arXiv.2509.13329
    fn move_items_multi(&mut self) -> SepStats {
        let master_sol = self.prob.save();

        // Anneal: rebuild the per-copy consistency-pressure map for the current
        // geometry and λ (master thread, O(n log n) on ~10^2 pieces — negligible
        // against the parallel move pass it precedes).
        let anneal_map: Option<Arc<AnnealMap>> = match (&self.grouped, self.anneal_lambda) {
            (Some(spec), l) if spec.flip.anneal && l > 0.0 => {
                Some(Arc::new(build_anneal_map(self, spec, l)))
            }
            _ => None,
        };

        // Define the parallel execution closure
        let mut separate_multi = || -> SepStats {
            self.workers.par_iter_mut().map(|worker| {
                // Sync the workers with the master
                worker.load(&master_sol, &self.ct, anneal_map.clone());
                // Let all of them run `move_items` with unique random orderings in which the items are moved
                worker.move_items()
            }).sum()
        };

        // Execute the parallel separation either using the local thread pool or the global one
        let sep_report = match self.thread_pool.as_mut() {
            Some(pool) => pool.install(&mut separate_multi),
            None => separate_multi(),
        };

        debug!("[MOD] optimizers w_o's: {:?}",self.workers.iter().map(|opt| opt.ct.get_total_weighted_loss()).collect_vec());

        // Check what run yielded the best solution (lowest collision quantification)
        let (best_sol, best_ct) = self.workers.iter_mut()
            .min_by_key(|opt| OrderedFloat(opt.ct.get_total_weighted_loss()))
            .map(|opt| (opt.prob.save(), &opt.ct))
            .unwrap();

        // Load this 'best' solution into the master, effectively throwing away all other work.
        self.prob.restore(&best_sol);
        self.ct = best_ct.clone();

        sep_report
    }

    pub fn rollback(&mut self, sol: &SPSolution, ots: Option<&CTSnapshot>) {
        debug_assert!(sol.strip_width() == self.prob.strip_width());
        self.prob.restore(sol);

        match ots {
            Some(ots) => {
                //if a snapshot of the tracker was provided, restore it
                self.ct.restore_but_keep_weights(ots, &self.prob.layout);
            }
            None => {
                //otherwise, rebuild it
                self.ct = CollisionTracker::new(&self.prob.layout);
            }
        }
    }

    pub fn move_item(&mut self, pk: PItemKey, d_transf: DTransformation) -> PItemKey {
        debug_assert!(tracker_matches_layout(&self.ct, &self.prob.layout));

        let item_id = self.prob.layout.placed_items[pk].item_id;

        let old_loss = self.ct.get_loss(pk);
        let old_weighted_loss = self.ct.get_weighted_loss(pk);

        //Remove the item from the problem
        self.prob.remove_item(pk);

        //Place the item again but with a new transformation
        let new_pk = self.prob.place_item(SPPlacement{d_transf,item_id});

        self.ct.register_item_move(&self.prob.layout, pk, new_pk);

        let new_loss = self.ct.get_loss(new_pk);
        let new_weighted_loss = self.ct.get_weighted_loss(new_pk);

        debug!("[MV] moved item {} from from l: {}, wl: {} to l+1: {}, wl+1: {}"
            ,item_id,FMT().fmt2(old_loss),FMT().fmt2(old_weighted_loss),FMT().fmt2(new_loss),FMT().fmt2(new_weighted_loss));

        debug_assert!(tracker_matches_layout(&self.ct, &self.prob.layout));

        new_pk
    }

    /// Proportional squeeze: every item's x-translation is scaled by
    /// new_width / old_width — many hairline overlaps across the whole strip,
    /// no seam, no item travels far. (See `ExplorationConfig::prop_shrink`.)
    pub fn change_strip_width_prop(&mut self, new_width: f32) {
        let ratio = new_width / self.prob.strip_width();
        let items = self.prob.layout.placed_items.iter()
            .filter(|(_, pi)| !pi_frozen(&self.frozen, pi))
            .map(|(k, pi)| (k, pi.d_transf))
            .collect_vec();
        for (pik, dtransf) in items {
            let t = dtransf.translation();
            self.move_item(pik, DTransformation::new(dtransf.rotation(), (t.0 * ratio, t.1)));
        }
        self.prob.change_strip_width(new_width);
        self.ct = CollisionTracker::new(&self.prob.layout);
        self.workers.iter_mut().for_each(|opt| {
            *opt = SeparatorWorker {
                instance: self.instance.clone(),
                prob: self.prob.clone(),
                ct: self.ct.clone(),
                rng: Xoshiro256PlusPlus::seed_from_u64(self.rng.random()),
                sample_config: self.config.sample_config,
                locked_items: self.locked_items.clone(),
                locks_active: self.locks_active.clone(),
                small: self.config.small,
                small_items: self.small_items.clone(),
                lane_centers: self.lane_centers.clone(),
                anneal: None,
                frozen: self.frozen.clone(),
            };
        });
    }

    pub fn change_strip_width(&mut self, new_width: f32, split_position: Option<f32>) {
        //if no split position is provided, use the center of the strip
        let split_position = split_position.unwrap_or(self.prob.strip_width() / 2.0);
        let delta = new_width - self.prob.strip_width();

        //shift all items right of the split position
        let items_to_shift = self.prob.layout.placed_items.iter()
            .filter(|(_, pi)| pi.shape.centroid().0 > split_position && !pi_frozen(&self.frozen, pi))
            .map(|(k, pi)| (k, pi.d_transf))
            .collect_vec();

        for (pik, dtransf) in items_to_shift {
            let existing_transf = dtransf.compose();
            let new_transf = existing_transf.translate((delta, 0.0));
            self.move_item(pik, new_transf.decompose());
        }

        self.prob.change_strip_width(new_width);

        //rebuild the collision tracker
        self.ct = CollisionTracker::new(&self.prob.layout);

        //rebuild the workers
        self.workers.iter_mut().for_each(|opt| {
            *opt = SeparatorWorker {
                instance: self.instance.clone(),
                prob: self.prob.clone(),
                ct: self.ct.clone(),
                rng: Xoshiro256PlusPlus::seed_from_u64(self.rng.random()),
                sample_config: self.config.sample_config,
                locked_items: self.locked_items.clone(),
                locks_active: self.locks_active.clone(),
                anneal: None,
                small: self.config.small,
                small_items: self.small_items.clone(),
                lane_centers: self.lane_centers.clone(),
                frozen: self.frozen.clone(),
            };
        });
        debug!("[SEP] changed strip width to {:.3}", new_width);
    }
}

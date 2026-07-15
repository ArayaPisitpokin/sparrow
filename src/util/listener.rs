use jagua_rs::probs::spp::entities::{SPInstance, SPSolution};

/// Trait for listeners that can receive solutions during the optimization process
pub trait SolutionListener {
    fn report(&mut self, report: ReportType, solution: &SPSolution, instance: &SPInstance);

    /// Called once at the end of each optimization phase with that phase's cumulative
    /// search statistics. Default: no-op. Fires twice per `optimize()` call — never on
    /// a hot path.
    fn on_search_stats(&mut self, _phase: StatsPhase, _stats: &SearchStats) {}
}

/// Which optimization phase a `SearchStats` report covers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StatsPhase {
    Exploration,
    Compression,
}

/// Cumulative search statistics for one optimization phase.
///
/// All counters are plain integers, owned and incremented by single-threaded code
/// (the master separation loop / the phase loops). `total_moves`/`total_evals` reuse
/// the per-`separate()` aggregation that already exists in the engine; nothing here
/// adds atomics, locks or timer reads to hot paths.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct SearchStats {
    /// Number of `separate()` calls performed.
    pub n_separate_calls: u64,
    /// Total item moves across all separate() calls (summed over workers).
    pub total_moves: u64,
    /// Total sample evaluations across all separate() calls (summed over workers).
    pub total_evals: u64,
    /// Exploration: successful strip shrinks. Compression: successful compressions.
    pub n_shrinks: u64,
    /// Number of disruption events (exploration only).
    pub n_disruptions: u64,
    /// Disruptions that were swap-two-large-items.
    pub n_swaps: u64,
    /// Disruptions that were group flips.
    pub n_flips: u64,
    /// Total garments flipped across all flip events.
    pub n_garments_flipped: u64,
    /// Sum of collision loss measured directly after each flip (repair burden proxy).
    pub flip_loss_injected: f32,
    /// Separations that reached feasibility directly after a flip disruption.
    pub post_flip_sep_success: u64,
    /// Separations that reached feasibility directly after a swap disruption.
    pub post_swap_sep_success: u64,
    /// Whether a grouped-orientation spec was active.
    pub grouped_active: bool,
    /// Wall-clock seconds of the phase (measured once at the phase boundary).
    pub phase_secs: f32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReportType {
    /// Report contains a feasible solution reached by the exploration phase.
    ExplFeas,
    /// Report contains an infeasible solution reached by the exploration phase.
    ExplInfeas,
    /// Report contains an intermediate solution from the exploration phase that is closer to feasibility than the previous one.
    ExplImproving,
    /// Report contains a feasible solution from the comparison phase.
    CmprFeas,
    /// Report contains the final solution
    Final
}

/// A dummy implementation of the `SolutionListener` trait that does nothing.
pub struct DummySolListener;

impl SolutionListener for DummySolListener {
    fn report(&mut self, _report: ReportType, _solution: &SPSolution, _instance: &SPInstance) {
        // Do nothing
    }
}

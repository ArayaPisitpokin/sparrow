//! Lévy-flight candidate sampler (experimental, env-gated).
//!
//! Sparrow's placement search draws candidates from two *uniform* distributions:
//! one inside the item's current bbox (very local) and one over the whole
//! container (global). That leaves a coverage gap at intermediate displacements
//! (a few piece-sizes away). A Lévy flight replaces the local sampler with a
//! single heavy-tailed displacement distribution centred on the current
//! translation: dense at small steps, thinning smoothly through the gap, with a
//! power-law tail reaching far.
//!
//! Steps use Mantegna's algorithm (`u / |v|^(1/beta)`), then are divided by the
//! *median* step for that beta (estimated once, deterministic seed) so the
//! median displacement equals `scale` (the item's smaller bbox dimension) for
//! every beta. Beta therefore controls only the tail shape, which is what makes
//! the beta≈2 (near-Gaussian) arm a clean "tail off" control.
//!
//! Env: `XEROW_EXP_LEVY_MODE` = off | focussed   (default off)
//!      `XEROW_EXP_LEVY_BETA` = f32              (default 1.5; valid (1,2))
//!      `XEROW_EXP_SAMPLE_MULT` = usize          (default 1; 1..=64; scales both sample counts)

use itertools::Itertools;
use jagua_rs::entities::Item;
use jagua_rs::geometry::geo_enums::RotationRange;
use jagua_rs::geometry::geo_traits::TransformableFrom;
use jagua_rs::geometry::primitives::Rect;
use jagua_rs::geometry::{DTransformation, Transformation};
use rand::prelude::IndexedRandom;
use rand::{Rng, RngExt, SeedableRng};
use std::f32::consts::PI;
use std::ops::Range;
use std::sync::OnceLock;

const ROT_N_SAMPLES: usize = 16;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LevyMode {
    Off,
    /// Replace the focussed (uniform-in-own-bbox) samples with Lévy samples.
    ReplaceFocussed,
}

#[derive(Clone, Copy, Debug)]
pub struct LevyCfg {
    pub mode: LevyMode,
    pub beta: f32,
    pub sigma_u: f32,
    /// Median of |mantegna step| for this beta; steps are divided by it.
    pub median_norm: f32,
    pub mult: usize,
}

static LEVY_CFG: OnceLock<LevyCfg> = OnceLock::new();

/// Read the experimental configuration from the environment exactly once.
pub fn levy_cfg() -> &'static LevyCfg {
    LEVY_CFG.get_or_init(|| {
        let mode = match std::env::var("XEROW_EXP_LEVY_MODE").ok().as_deref() {
            Some("focussed") => LevyMode::ReplaceFocussed,
            _ => LevyMode::Off,
        };
        let beta = std::env::var("XEROW_EXP_LEVY_BETA").ok()
            .and_then(|s| s.parse::<f32>().ok())
            .filter(|b| *b > 1.0 && *b < 2.0)
            .unwrap_or(1.5);
        let mult = std::env::var("XEROW_EXP_SAMPLE_MULT").ok()
            .and_then(|s| s.parse::<usize>().ok())
            .filter(|m| *m >= 1 && *m <= 64)
            .unwrap_or(1);
        let sigma_u = mantegna_sigma_u(beta);
        let median_norm = estimate_median_step(beta, sigma_u);
        let cfg = LevyCfg { mode, beta, sigma_u, median_norm, mult };
        if mode != LevyMode::Off || mult != 1 {
            eprintln!("[xerow-exp] LEVY mode={:?} beta={} sigma_u={:.4} median_norm={:.4} sample_mult={}",
                      mode, beta, sigma_u, median_norm, mult);
        }
        cfg
    })
}

/// A sampler producing Lévy-distributed displacements from a centre translation.
#[derive(Clone, Debug)]
pub struct LevySampler {
    center: (f32, f32),
    scale: f32,
    beta: f32,
    sigma_u: f32,
    median_norm: f32,
    rot_entries: Vec<RotEntry>,
}

#[derive(Clone, Debug)]
struct RotEntry {
    r: f32,
    x_range: Range<f32>,
    y_range: Range<f32>,
}

impl LevySampler {
    /// `center`: the item's current translation. `scale`: characteristic length
    /// (median displacement). Rotation handling mirrors `UniformBBoxSampler`.
    pub fn new(center: (f32, f32), item: &Item, container_bbox: Rect, rot_override: Option<f32>,
               scale: f32, cfg: &LevyCfg) -> Option<Self> {
        let locked;
        let rotations = match rot_override {
            Some(r) => { locked = vec![r]; &locked }
            None => match &item.allowed_rotation {
                RotationRange::None => &vec![0.0],
                RotationRange::Discrete(r) => r,
                RotationRange::Continuous => {
                    let step = (2.0 * PI) / ROT_N_SAMPLES as f32;
                    &(0..ROT_N_SAMPLES).map(|i| i as f32 * step).collect_vec()
                }
            },
        };
        let mut shape_buffer = item.shape_cd.as_ref().clone();
        let rot_entries = rotations.iter().filter_map(|&r| {
            let bb = shape_buffer.transform_from(item.shape_cd.as_ref(), &Transformation::from_rotation(r)).bbox;
            let x_range = (container_bbox.x_min - bb.x_min)..(container_bbox.x_max - bb.x_max);
            let y_range = (container_bbox.y_min - bb.y_min)..(container_bbox.y_max - bb.y_max);
            if x_range.is_empty() || y_range.is_empty() { None } else { Some(RotEntry { r, x_range, y_range }) }
        }).collect_vec();
        if rot_entries.is_empty() {
            return None;
        }
        Some(Self { center, scale, beta: cfg.beta, sigma_u: cfg.sigma_u, median_norm: cfg.median_norm, rot_entries })
    }

    pub fn sample(&self, rng: &mut impl Rng) -> DTransformation {
        let e = self.rot_entries.choose(rng).unwrap();
        let step = levy_step(rng, self.beta, self.sigma_u).abs() / self.median_norm;
        let r = self.scale * step;
        let theta = rng.random::<f32>() * 2.0 * PI;
        let mut x = self.center.0 + r * theta.cos();
        let mut y = self.center.1 + r * theta.sin();
        // A jump that overshoots the container-valid range is replaced by a
        // uniform draw inside it (same half-open range the uniform sampler uses).
        // Never clamp onto the exact boundary: an exact-edge translation can
        // round outside the container on export/re-import and fail the external
        // feasibility check while sparrow's own engine accepted it.
        if !(x >= e.x_range.start && x < e.x_range.end) {
            x = rng.random_range(e.x_range.clone());
        }
        if !(y >= e.y_range.start && y < e.y_range.end) {
            y = rng.random_range(e.y_range.clone());
        }
        DTransformation::new(e.r, (x, y))
    }
}

/// Standard normal via Box–Muller.
fn gauss(rng: &mut impl Rng) -> f32 {
    let u1 = rng.random::<f32>().max(1e-12);
    let u2 = rng.random::<f32>();
    (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
}

/// One Mantegna Lévy step (signed).
fn levy_step(rng: &mut impl Rng, beta: f32, sigma_u: f32) -> f32 {
    let u = gauss(rng) * sigma_u;
    let v = gauss(rng).abs().max(1e-12);
    u / v.powf(1.0 / beta)
}

fn mantegna_sigma_u(beta: f32) -> f32 {
    let num = gamma(1.0 + beta) * (PI * beta / 2.0).sin();
    let den = gamma((1.0 + beta) / 2.0) * beta * 2f32.powf((beta - 1.0) / 2.0);
    (num / den).powf(1.0 / beta)
}

/// Median of |step| for this beta, from a fixed-seed Monte Carlo (deterministic).
fn estimate_median_step(beta: f32, sigma_u: f32) -> f32 {
    let mut rng = rand::rngs::SmallRng::seed_from_u64(0x5EED_1EE7);
    let mut v: Vec<f32> = (0..20_000).map(|_| levy_step(&mut rng, beta, sigma_u).abs()).collect();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2].max(1e-9)
}

/// Lanczos approximation of the gamma function (g = 7, n = 9).
fn gamma(x: f32) -> f32 {
    const G: f64 = 7.0;
    const C: [f64; 9] = [
        0.999_999_999_999_809_93, 676.520_368_121_885_1, -1_259.139_216_722_402_8,
        771.323_428_777_653_13, -176.615_029_162_140_59, 12.507_343_278_686_905,
        -0.138_571_095_265_720_12, 9.984_369_578_019_571_6e-6, 1.505_632_735_149_311_6e-7,
    ];
    let x = x as f64;
    if x < 0.5 {
        (std::f64::consts::PI / ((std::f64::consts::PI * x).sin() * gamma((1.0 - x) as f32) as f64)) as f32
    } else {
        let x = x - 1.0;
        let mut a = C[0];
        let t = x + G + 0.5;
        for (i, c) in C.iter().enumerate().skip(1) {
            a += c / (x + i as f64);
        }
        ((2.0 * std::f64::consts::PI).sqrt() * t.powf(x + 0.5) * (-t).exp() * a) as f32
    }
}

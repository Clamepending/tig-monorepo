use serde_json::{Map, Value};

#[derive(Clone, Copy)]
pub struct Params {
    pub n_perturbation_rounds: usize,
    pub perturbation_strength_base: usize,
    pub extra_starts: usize,
    pub max_frontier_swaps_override: Option<usize>,
    pub dp_passes_multiplier: usize,
}

impl Params {
    pub fn initialize(h: &Option<Map<String, Value>>) -> Self {
        // v5 = same skeleton as v4 but burns more fuel on exploration.
        // v4 finishes n_items=1000,budget=25 in ~3 s (n=5 wall=15s) at fuel
        // 5e12 — way under budget — so doubling perturbation rounds + adding
        // extra starts is essentially free.
        let mut p = Self {
            n_perturbation_rounds: 30,
            perturbation_strength_base: 4,
            extra_starts: 4,
            max_frontier_swaps_override: None,
            dp_passes_multiplier: 1,
        };
        if let Some(m) = h {
            if let Some(v) = m.get("n_perturbation_rounds").and_then(|x| x.as_u64()) {
                p.n_perturbation_rounds = v as usize;
            }
            if let Some(v) = m.get("perturbation_strength_base").and_then(|x| x.as_u64()) {
                p.perturbation_strength_base = v as usize;
            }
            if let Some(v) = m.get("extra_starts").and_then(|x| x.as_u64()) {
                p.extra_starts = v as usize;
            }
            if let Some(v) = m.get("dp_passes_multiplier").and_then(|x| x.as_u64()) {
                p.dp_passes_multiplier = v as usize;
            }
        }
        p
    }
}

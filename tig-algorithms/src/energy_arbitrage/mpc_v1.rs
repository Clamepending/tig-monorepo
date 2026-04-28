// Per-node forecast-percentile policy for c008 energy_arbitrage.
//
// At each step t and for each battery b at node n, compute the 24-step DA-price
// distribution at node n starting from t. If the *current* RT price at node n
// is in the lower forecast quartile, charge; if in the upper quartile,
// discharge. Otherwise idle. Then enforce flow feasibility via the same
// soften-most-violated-line loop as the on-platform greedy baseline.
//
// Improvements over the on-platform greedy baseline:
//  * Per-node DA price (greedy uses node 0 as a proxy across all batteries).
//  * RT price as the trigger signal, not DA — RT carries the volatility +
//    congestion premium + jump term that greedy misses.
//  * Forward window of 24 steps (6 h) vs greedy's 3 h.
//  * Quartile threshold (data-driven) vs fixed $5 (track-blind).
//
// Local-only validation. Submission cost 10 TIG; do not submit on-chain
// without (a) beating the on-platform `min_active_quality=1` threshold beyond
// noise on n>=5 nonces per scenario and (b) human-in-loop wallet step.

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use tig_challenges::energy_arbitrage::*;

const FORECAST_HORIZON: usize = 24; // 6 h of 15-minute steps
const MAX_FLOW_ADJUST_ITERS: usize = 64;
const GLOBAL_SCALE_BSEARCH_ITERS: usize = 32;
const EPS: f64 = 1e-12;
const EPS_FLOW: f64 = 1e-6;

#[derive(Serialize, Deserialize)]
pub struct Hyperparameters {
    pub low_quantile: Option<f64>,
    pub high_quantile: Option<f64>,
    pub horizon: Option<usize>,
}

pub fn help() {
    println!("MPC-v1 — per-node DA-price-quantile threshold policy with flow-feasibility softener");
    println!("  hyperparameters: low_quantile (default 0.30), high_quantile (default 0.70), horizon (default 24)");
}

pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
) -> Result<()> {
    let hp: Hyperparameters = match hyperparameters {
        Some(m) => serde_json::from_value::<Hyperparameters>(Value::Object(m.clone()))
            .unwrap_or(Hyperparameters {
                low_quantile: None,
                high_quantile: None,
                horizon: None,
            }),
        None => Hyperparameters {
            low_quantile: None,
            high_quantile: None,
            horizon: None,
        },
    };
    let low_q = hp.low_quantile.unwrap_or(0.30);
    let high_q = hp.high_quantile.unwrap_or(0.70);
    let horizon = hp.horizon.unwrap_or(FORECAST_HORIZON);

    let policy = move |challenge: &Challenge, state: &State| -> Result<Vec<f64>> {
        compute_action(challenge, state, low_q, high_q, horizon)
    };
    let solution = challenge.grid_optimize(&policy)?;
    save_solution(&solution)?;
    Ok(())
}

fn compute_action(
    challenge: &Challenge,
    state: &State,
    low_q: f64,
    high_q: f64,
    horizon: usize,
) -> Result<Vec<f64>> {
    let t = state.time_step;
    let end = (t + horizon).min(challenge.num_steps);
    let da_prices = &challenge.market.day_ahead_prices;

    let mut action = vec![0.0; challenge.num_batteries];

    for (b, battery) in challenge.batteries.iter().enumerate() {
        let node = battery.node;
        let (lo_bound, hi_bound) = state.action_bounds[b];
        let rt_now = state.rt_prices[node];

        // Build per-node forecast window from DA prices.
        let mut window: Vec<f64> = Vec::with_capacity(horizon);
        for tau in t..end {
            window.push(da_prices[tau][node]);
        }
        if window.is_empty() {
            continue;
        }
        // Quantiles via in-place sort — cheap (horizon <= 24).
        let mut sorted = window.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let lo_idx = ((sorted.len() as f64 - 1.0) * low_q).round() as usize;
        let hi_idx = ((sorted.len() as f64 - 1.0) * high_q).round() as usize;
        let p_low = sorted[lo_idx];
        let p_high = sorted[hi_idx];

        if rt_now <= p_low {
            // Charge fully (lo_bound is negative or zero by convention).
            action[b] = lo_bound;
        } else if rt_now >= p_high {
            // Discharge fully.
            action[b] = hi_bound;
        }
    }

    enforce_flow_feasibility(challenge, state, action)
}

#[derive(Clone, Copy)]
struct Violation {
    line: usize,
    flow: f64,
    amount: f64,
}

fn compute_flows(challenge: &Challenge, state: &State, action: &[f64]) -> Vec<f64> {
    let injections = challenge.compute_total_injections(state, action);
    (0..challenge.network.num_lines)
        .map(|l| {
            (0..challenge.network.num_nodes)
                .map(|k| challenge.network.ptdf[l][k] * injections[k])
                .sum::<f64>()
        })
        .collect()
}

fn most_violated_line(challenge: &Challenge, flows: &[f64]) -> Option<Violation> {
    let mut best: Option<Violation> = None;
    for (l, &flow) in flows.iter().enumerate() {
        let limit = challenge.network.flow_limits[l];
        let violation = flow.abs() - limit;
        if violation > EPS_FLOW * limit {
            let candidate = Violation {
                line: l,
                flow,
                amount: violation,
            };
            match best {
                Some(current) if candidate.amount <= current.amount => {}
                _ => best = Some(candidate),
            }
        }
    }
    best
}

fn is_flow_feasible(challenge: &Challenge, state: &State, action: &[f64]) -> bool {
    let flows = compute_flows(challenge, state, action);
    most_violated_line(challenge, &flows).is_none()
}

fn soften_most_violated_line(
    challenge: &Challenge,
    violation: Violation,
    action: &mut [f64],
) -> bool {
    let line = violation.line;
    let signed_direction = violation.flow.signum();
    if signed_direction.abs() <= EPS {
        return false;
    }

    let mut worsening_indices = Vec::new();
    let mut worsening_strength = 0.0;
    for (i, battery) in challenge.batteries.iter().enumerate() {
        let contribution = challenge.network.ptdf[line][battery.node] * action[i];
        let signed_contribution = signed_direction * contribution;
        if signed_contribution > EPS {
            worsening_strength += signed_contribution;
            worsening_indices.push(i);
        }
    }

    if worsening_indices.is_empty() || worsening_strength <= EPS {
        return false;
    }

    let keep = (1.0 - violation.amount / worsening_strength).clamp(0.0, 1.0);
    if (1.0 - keep).abs() <= EPS {
        return false;
    }
    for i in worsening_indices {
        action[i] *= keep;
    }
    true
}

fn enforce_flow_feasibility(
    challenge: &Challenge,
    state: &State,
    mut action: Vec<f64>,
) -> Result<Vec<f64>> {
    for _ in 0..MAX_FLOW_ADJUST_ITERS {
        let flows = compute_flows(challenge, state, &action);
        let Some(violation) = most_violated_line(challenge, &flows) else {
            return Ok(action);
        };
        if !soften_most_violated_line(challenge, violation, &mut action) {
            break;
        }
    }

    if is_flow_feasible(challenge, state, &action) {
        return Ok(action);
    }

    let zero = vec![0.0; action.len()];
    if !is_flow_feasible(challenge, state, &zero) {
        return Err(anyhow!(
            "MPC-v1 fallback failed: grid is infeasible even with zero battery actions"
        ));
    }

    let base = action;
    let mut low = 0.0;
    let mut high = 1.0;
    for _ in 0..GLOBAL_SCALE_BSEARCH_ITERS {
        let mid = 0.5 * (low + high);
        let scaled: Vec<f64> = base.iter().map(|u| mid * u).collect();
        if is_flow_feasible(challenge, state, &scaled) {
            low = mid;
        } else {
            high = mid;
        }
    }
    Ok(base.into_iter().map(|u| low * u).collect())
}

// probSAT v2 for c001 satisfiability.
//
// Improvements over v1:
//  * Jeroslow-Wang initialization: pick each variable's polarity to match the
//    sign that occurs more often in the formula (3-SAT collapses JW to
//    majority).
//  * Schöning hybrid step: with probability wp pick a uniform random literal
//    in the unsat clause (random walk), else use probSAT formula.
//  * Tabu list: prevent re-flipping a variable within last K flips.

use anyhow::Result;
use rand::{rngs::SmallRng, Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use tig_challenges::satisfiability::*;

#[derive(Serialize, Deserialize)]
pub struct Hyperparameters {
    pub c_b: Option<f64>,
    pub eps: Option<f64>,
    pub luby_base: Option<u64>,
    pub max_total_flips: Option<u64>,
    pub wp_random_walk: Option<f64>,
    pub tabu_size: Option<usize>,
}

pub fn help() {
    println!("probSAT v2 — JW init + Schöning random-walk hybrid + tabu list + Luby restarts");
}

pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
) -> Result<()> {
    let n = challenge.num_variables;
    let _ = save_solution(&Solution {
        variables: vec![false; n],
    });

    let hp: Hyperparameters = match hyperparameters {
        Some(m) => serde_json::from_value(Value::Object(m.clone())).unwrap_or(Hyperparameters {
            c_b: None,
            eps: None,
            luby_base: None,
            max_total_flips: None,
            wp_random_walk: None,
            tabu_size: None,
        }),
        None => Hyperparameters {
            c_b: None,
            eps: None,
            luby_base: None,
            max_total_flips: None,
            wp_random_walk: None,
            tabu_size: None,
        },
    };
    // Defaults: cap at 1B flips → ~500–1500s wallclock per nonce on n=5000.
    // probSAT's LLVM-IR-instrumented fuel decrement is sparser than sat_vanguard's,
    // so without an explicit flip cap the algorithm will run far longer than sat_vanguard
    // and never hit the 5e12 fuel ceiling. Cap = budget control.
    let c_b = hp.c_b.unwrap_or(2.06);
    let eps = hp.eps.unwrap_or(1.0);
    let luby_base = hp.luby_base.unwrap_or(4096);
    let max_total_flips = hp.max_total_flips.unwrap_or(1_000_000_000);
    let wp = hp.wp_random_walk.unwrap_or(0.05);
    let tabu_size = hp.tabu_size.unwrap_or(8);

    let mut clauses: Vec<[i32; 3]> = Vec::new();
    for cl in &challenge.clauses {
        let (a, b, c) = (cl[0], cl[1], cl[2]);
        if a == -b || a == -c || b == -c {
            continue;
        }
        clauses.push([a, b, c]);
    }
    let nc = clauses.len();

    let mut pos_count = vec![0i32; n];
    let mut neg_count = vec![0i32; n];
    let mut pos_clauses: Vec<Vec<u32>> = vec![Vec::new(); n];
    let mut neg_clauses: Vec<Vec<u32>> = vec![Vec::new(); n];
    for (ci, cl) in clauses.iter().enumerate() {
        for &lit in cl.iter() {
            let v = (lit.unsigned_abs() as usize) - 1;
            if lit > 0 {
                pos_count[v] += 1;
                pos_clauses[v].push(ci as u32);
            } else {
                neg_count[v] += 1;
                neg_clauses[v].push(ci as u32);
            }
        }
    }

    let mut rng = SmallRng::from_seed(challenge.seed);

    let jw_init: Vec<bool> = (0..n)
        .map(|v| {
            if pos_count[v] > neg_count[v] {
                true
            } else if pos_count[v] < neg_count[v] {
                false
            } else {
                rng.r#gen::<bool>()
            }
        })
        .collect();

    fn luby(mut k: u64) -> u64 {
        let mut size = 1u64;
        let mut seq = 1u64;
        while size < k + 1 {
            seq = size;
            size = (size * 2) + 1;
        }
        while size - 1 != k {
            size = (size - 1) >> 1;
            if size == 0 {
                break;
            }
            if k >= size {
                k -= size;
                size = (size << 1) + 1;
                seq <<= 1;
            }
        }
        seq.max(1)
    }

    let mut total_flips: u64 = 0;
    let mut restart_idx: u64 = 0;

    let mut assignment: Vec<bool> = jw_init.clone();
    let mut clause_sat_lits: Vec<u8> = vec![0; nc];
    let mut unsat_clauses: Vec<u32> = Vec::with_capacity(nc);
    let mut unsat_pos: Vec<i32> = vec![-1; nc];
    let mut tabu: Vec<i64> = vec![-1; n];
    let tabu_size = tabu_size as i64;

    'outer: while total_flips < max_total_flips {
        if restart_idx == 0 {
            assignment.copy_from_slice(&jw_init);
        } else {
            for v in 0..n {
                assignment[v] = rng.r#gen::<bool>();
            }
        }

        clause_sat_lits.fill(0);
        unsat_clauses.clear();
        unsat_pos.fill(-1);
        tabu.fill(-1);
        for (ci, cl) in clauses.iter().enumerate() {
            let mut sat = 0u8;
            for &lit in cl.iter() {
                let v = (lit.unsigned_abs() as usize) - 1;
                let val = assignment[v];
                if (lit > 0 && val) || (lit < 0 && !val) {
                    sat += 1;
                }
            }
            clause_sat_lits[ci] = sat;
            if sat == 0 {
                unsat_pos[ci] = unsat_clauses.len() as i32;
                unsat_clauses.push(ci as u32);
            }
        }

        let restart_budget = luby_base.saturating_mul(luby(restart_idx));
        restart_idx += 1;

        let mut flips_in_restart: u64 = 0;
        while flips_in_restart < restart_budget && total_flips < max_total_flips {
            if unsat_clauses.is_empty() {
                let _ = save_solution(&Solution {
                    variables: assignment.clone(),
                });
                return Ok(());
            }

            let pick = rng.gen_range(0..unsat_clauses.len());
            let ci = unsat_clauses[pick] as usize;
            let cl = &clauses[ci];

            let flip_v: usize;

            if rng.r#gen::<f64>() < wp {
                let pick_lit_idx = rng.gen_range(0..3usize);
                flip_v = (cl[pick_lit_idx].unsigned_abs() as usize) - 1;
            } else {
                let mut probs: [f64; 3] = [0.0; 3];
                let cur_step = total_flips as i64;
                for (i, &lit) in cl.iter().enumerate() {
                    let v = (lit.unsigned_abs() as usize) - 1;
                    if tabu[v] >= 0 && cur_step - tabu[v] < tabu_size {
                        probs[i] = 0.0;
                        continue;
                    }
                    let cur_val = assignment[v];
                    let candidates: &Vec<u32> = if cur_val {
                        &pos_clauses[v]
                    } else {
                        &neg_clauses[v]
                    };
                    let mut brk: u32 = 0;
                    for &cj in candidates {
                        if clause_sat_lits[cj as usize] == 1 {
                            brk += 1;
                        }
                    }
                    probs[i] = (eps + brk as f64).powf(-c_b);
                }

                let total: f64 = probs[0] + probs[1] + probs[2];
                if total <= 0.0 {
                    let pick_lit_idx = rng.gen_range(0..3usize);
                    flip_v = (cl[pick_lit_idx].unsigned_abs() as usize) - 1;
                } else {
                    let r: f64 = rng.r#gen::<f64>() * total;
                    let mut idx = 2usize;
                    let mut acc = 0.0;
                    for i in 0..3 {
                        acc += probs[i];
                        if r <= acc {
                            idx = i;
                            break;
                        }
                    }
                    flip_v = (cl[idx].unsigned_abs() as usize) - 1;
                }
            }

            tabu[flip_v] = total_flips as i64;
            let new_val = !assignment[flip_v];
            assignment[flip_v] = new_val;

            let lost = if new_val {
                &neg_clauses[flip_v]
            } else {
                &pos_clauses[flip_v]
            };
            for &cj_u in lost {
                let cj = cj_u as usize;
                clause_sat_lits[cj] -= 1;
                if clause_sat_lits[cj] == 0 {
                    unsat_pos[cj] = unsat_clauses.len() as i32;
                    unsat_clauses.push(cj_u);
                }
            }
            let gained = if new_val {
                &pos_clauses[flip_v]
            } else {
                &neg_clauses[flip_v]
            };
            for &cj_u in gained {
                let cj = cj_u as usize;
                clause_sat_lits[cj] += 1;
                if clause_sat_lits[cj] == 1 {
                    let pos = unsat_pos[cj] as usize;
                    let last = unsat_clauses.len() - 1;
                    if pos != last {
                        unsat_clauses[pos] = unsat_clauses[last];
                        unsat_pos[unsat_clauses[pos] as usize] = pos as i32;
                    }
                    unsat_clauses.pop();
                    unsat_pos[cj] = -1;
                }
            }

            flips_in_restart += 1;
            total_flips += 1;
        }

        if total_flips >= max_total_flips {
            break 'outer;
        }
    }

    Ok(())
}

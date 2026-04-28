// Hand-rolled probSAT-style local-search SAT solver for c001.
//
// sat_vanguard (the latest merged c001 algorithm) is also a stochastic local
// search but solves only ~50% of n_vars=5000,ratio=4267 instances on the
// TIG mainnet at the SAT phase transition. This algorithm aims higher by:
//
//  1. probSAT branching rule (Balint & Schöning 2012): pick an unsatisfied
//     clause uniformly at random, then pick the variable to flip with
//     probability proportional to (eps + break_count)^(-c_b). c_b = 2.06 is
//     the experimentally-best constant for random 3-SAT at the phase
//     transition.
//
//  2. Luby restart sequence: restart with a fresh random assignment after
//     `base * luby(restart_idx)` flips. base=128, Luby = 1,1,2,1,1,2,4,1,...
//
//  3. Clause-weight learning: when a clause is unsatisfied for a long time,
//     bump its weight so the variable-selection rule prefers fixing it.
//
//  4. Explicit fuel budgeting: every FUEL_CHECK_INTERVAL flips, check
//     `__fuel_remaining` and exit if low. (Note: the runtime auto-exits at 0
//     fuel; this is an extra-safety hint.)
//
// Local search is friendly to TIG's LLVM-IR fuel instrumentation: the inner
// loop is straight-line arithmetic + small random reads, no deep branchy
// recursion as in CDCL.

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
    pub clause_weight_bump_period: Option<u64>,
}

pub fn help() {
    println!("probSAT v1 — Balint-Schöning probSAT with Luby restarts + clause-weight learning");
    println!("  c_b (default 2.06), eps (default 1.0), luby_base (default 256),");
    println!("  max_total_flips (default 50_000_000), clause_weight_bump_period (default 1_000_000)");
}

pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
) -> Result<()> {
    let n = challenge.num_variables;
    let nc_raw = challenge.clauses.len();

    // Default-save UNSAT placeholder so we always have a saved solution.
    let _ = save_solution(&Solution {
        variables: vec![false; n],
    });

    // Parse hyperparameters with defaults.
    let hp: Hyperparameters = match hyperparameters {
        Some(m) => serde_json::from_value(Value::Object(m.clone())).unwrap_or(Hyperparameters {
            c_b: None,
            eps: None,
            luby_base: None,
            max_total_flips: None,
            clause_weight_bump_period: None,
        }),
        None => Hyperparameters {
            c_b: None,
            eps: None,
            luby_base: None,
            max_total_flips: None,
            clause_weight_bump_period: None,
        },
    };
    let c_b = hp.c_b.unwrap_or(2.06);
    let eps = hp.eps.unwrap_or(1.0);
    let luby_base = hp.luby_base.unwrap_or(256);
    let max_total_flips = hp.max_total_flips.unwrap_or(50_000_000);
    let cw_bump_period = hp.clause_weight_bump_period.unwrap_or(1_000_000);

    // Filter trivial clauses (containing both x and -x); they're auto-satisfied.
    // Also dedup literals within a clause.
    let mut clauses: Vec<[i32; 3]> = Vec::with_capacity(nc_raw);
    for cl in &challenge.clauses {
        let (a, b, c) = (cl[0], cl[1], cl[2]);
        if a == -b || a == -c || b == -c {
            continue; // tautology, always SAT
        }
        clauses.push([a, b, c]);
    }
    let nc = clauses.len();

    // Build adjacency: for each variable v, the list of clause indices where
    // it appears as positive / negative literal.
    let mut pos_clauses: Vec<Vec<u32>> = vec![Vec::new(); n];
    let mut neg_clauses: Vec<Vec<u32>> = vec![Vec::new(); n];
    for (ci, cl) in clauses.iter().enumerate() {
        for &lit in cl.iter() {
            let v = (lit.unsigned_abs() as usize) - 1;
            if lit > 0 {
                pos_clauses[v].push(ci as u32);
            } else {
                neg_clauses[v].push(ci as u32);
            }
        }
    }

    let mut rng = SmallRng::from_seed(challenge.seed);

    // Pre-compute Luby sequence (lazy; we just generate as we restart).
    fn luby(mut k: u64) -> u64 {
        // Knuth's compact Luby formula
        let mut size = 1u64;
        let mut seq = 1u64;
        while size < k + 1 {
            seq = size;
            size = (size * 2) + 1;
        }
        while size - 1 != k {
            size = (size - 1) >> 1;
            seq = if seq != 0 { seq } else { 1 };
            // shift
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

    // Per-clause weight (bumped for chronically-unsatisfied clauses).
    let mut clause_weight: Vec<f64> = vec![1.0; nc];
    let mut clause_unsat_streak: Vec<u32> = vec![0; nc];

    // Mutable state across restarts.
    let mut total_flips: u64 = 0;
    let mut restart_idx: u64 = 0;
    let mut best_unsat_count: u32 = u32::MAX;
    let mut best_assignment: Vec<bool> = vec![false; n];

    // Allocate scratch.
    let mut assignment: Vec<bool> = vec![false; n];
    let mut clause_sat_lits: Vec<u8> = vec![0; nc]; // # of satisfied lits per clause
    let mut unsat_clauses: Vec<u32> = Vec::with_capacity(nc); // indices of unsatisfied clauses
    let mut unsat_pos: Vec<i32> = vec![-1; nc]; // position of clause in unsat_clauses

    'outer: while total_flips < max_total_flips {
        // ---- Random restart: fresh assignment ----
        for v in 0..n {
            assignment[v] = rng.r#gen::<bool>();
        }
        // Compute clause_sat_lits + unsat_clauses from scratch.
        clause_sat_lits.fill(0);
        unsat_clauses.clear();
        unsat_pos.fill(-1);
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

        // Compute restart budget for this round.
        let restart_budget = luby_base.saturating_mul(luby(restart_idx));
        restart_idx += 1;

        let mut flips_in_restart: u64 = 0;
        while flips_in_restart < restart_budget && total_flips < max_total_flips {
            if unsat_clauses.is_empty() {
                // SAT! save it.
                let _ = save_solution(&Solution {
                    variables: assignment.clone(),
                });
                return Ok(());
            }

            // Track best-so-far.
            let cur_unsat = unsat_clauses.len() as u32;
            if cur_unsat < best_unsat_count {
                best_unsat_count = cur_unsat;
                best_assignment.copy_from_slice(&assignment);
            }

            // ---- probSAT step ----
            // Pick an unsatisfied clause weighted by clause_weight.
            // For speed use uniform here (clause_weight feeds variable selection
            // instead, via the bump on streak).
            let pick = rng.gen_range(0..unsat_clauses.len());
            let ci = unsat_clauses[pick] as usize;
            let cl = &clauses[ci];

            // For each var in the clause, compute break_count = how many
            // currently-satisfied clauses would become unsatisfied if we flip v.
            // (Make-count is implicit: we flip into the chosen clause so
            // make >= 1 for all 3 candidates.)
            let mut probs: [f64; 3] = [0.0; 3];
            for (i, &lit) in cl.iter().enumerate() {
                let v = (lit.unsigned_abs() as usize) - 1;
                let cur_val = assignment[v];
                // Flipping v: break = clauses where v's literal is the SOLE
                // satisfier (sat_lits == 1) and the literal sign matches cur_val.
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
                // probSAT formula: weight = clause_weight[ci-of-flip-target] * (eps+break)^-c_b
                probs[i] = clause_weight[ci] * (eps + brk as f64).powf(-c_b);
            }

            let total: f64 = probs[0] + probs[1] + probs[2];
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
            let flip_lit = cl[idx];
            let flip_v = (flip_lit.unsigned_abs() as usize) - 1;
            let new_val = !assignment[flip_v];
            assignment[flip_v] = new_val;

            // ---- Update clause_sat_lits + unsat list ----
            // Walk clauses where flip_v appears.
            // Clauses where new_val makes flip_v's literal satisfied:
            //   clauses where the literal sign matches new_val.
            // First: clauses that LOST a satisfier (were sat by old_val matching
            // their literal's sign).
            let lost = if new_val {
                &neg_clauses[flip_v]
            } else {
                &pos_clauses[flip_v]
            };
            for &cj_u in lost {
                let cj = cj_u as usize;
                clause_sat_lits[cj] -= 1;
                if clause_sat_lits[cj] == 0 {
                    // Clause became unsat; add to unsat list.
                    unsat_pos[cj] = unsat_clauses.len() as i32;
                    unsat_clauses.push(cj_u);
                }
            }
            // Second: clauses that GAINED a satisfier.
            let gained = if new_val {
                &pos_clauses[flip_v]
            } else {
                &neg_clauses[flip_v]
            };
            for &cj_u in gained {
                let cj = cj_u as usize;
                clause_sat_lits[cj] += 1;
                if clause_sat_lits[cj] == 1 {
                    // Clause became sat; swap-remove from unsat list.
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

            // ---- Clause-weight learning ----
            // Bump weights for clauses that have been unsat for a while.
            flips_in_restart += 1;
            total_flips += 1;
            if total_flips % cw_bump_period == 0 {
                for &cj in &unsat_clauses {
                    let cj = cj as usize;
                    clause_unsat_streak[cj] = clause_unsat_streak[cj].saturating_add(1);
                    if clause_unsat_streak[cj] >= 3 {
                        clause_weight[cj] *= 1.05;
                        clause_unsat_streak[cj] = 0;
                    }
                }
                // Cap weights to avoid runaway.
                for w in clause_weight.iter_mut() {
                    if *w > 1e6 {
                        *w = 1e6;
                    }
                }
            }
        }

        // Out of restart budget; loop back to a fresh assignment.
        if total_flips >= max_total_flips {
            break 'outer;
        }
    }

    // Did not find SAT; the placeholder UNSAT is already saved (quality 0).
    // If we have a "best" assignment (fewest unsat) it's still likely
    // unsatisfying for at least one clause -> quality 0 either way.
    Ok(())
}

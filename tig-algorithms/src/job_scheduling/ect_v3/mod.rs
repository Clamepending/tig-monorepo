// Multi-rule list scheduler for c007 job_scheduling.
//
// v2 used a single dispatch rule (earliest-ready + LRPT tiebreak) and got
// makespan 4% worse than baseline. v3 runs N different priority rules in
// sequence, picks the schedule with smallest makespan. Adds modest randomized
// shuffles for further diversification while there's fuel budget.
//
// Priority rules tried (each is a (priority_fn, machine_pick) pair):
//  - earliest-ready + LRPT (longest remaining processing time)
//  - earliest-ready + SRPT (shortest remaining)
//  - earliest-ready + most-eligible-machines-first (least flexible jobs first)
//  - earliest-ready + LPT (longest current operation)
//  - earliest-ready + random-tiebreak
//
// Machine pick is always ECT (earliest finish time among eligible).

use anyhow::{anyhow, Result};
use rand::{rngs::SmallRng, Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use tig_challenges::job_scheduling::*;

#[derive(Serialize, Deserialize)]
pub struct Hyperparameters {
    pub n_random_passes: Option<usize>,
}

pub fn help() {
    println!("ECT v3 — multi-rule list scheduler + random-tiebreak passes; pick min makespan.");
}

#[derive(Clone, Copy)]
enum Priority {
    Lrpt,
    Srpt,
    LeastFlexible,
    Lpt,
    Random(u64),
}

pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
) -> Result<()> {
    let hp: Hyperparameters = match hyperparameters {
        Some(m) => serde_json::from_value(Value::Object(m.clone()))
            .unwrap_or(Hyperparameters {
                n_random_passes: None,
            }),
        None => Hyperparameters {
            n_random_passes: None,
        },
    };
    let n_random_passes = hp.n_random_passes.unwrap_or(64);

    let n_jobs = challenge.num_jobs;
    let mut job_to_product: Vec<usize> = Vec::with_capacity(n_jobs);
    for (product, &num_in_product) in challenge.jobs_per_product.iter().enumerate() {
        for _ in 0..num_in_product {
            job_to_product.push(product);
        }
    }
    if job_to_product.len() != n_jobs {
        return Err(anyhow!("jobs_per_product sum != num_jobs"));
    }

    // Pre-compute remaining-work suffix per (job, op).
    let remaining_work: Vec<Vec<u32>> = (0..n_jobs)
        .map(|j| {
            let p = job_to_product[j];
            let ops = &challenge.product_processing_times[p];
            let mut suf = vec![0u32; ops.len() + 1];
            for k in (0..ops.len()).rev() {
                let min_pt = ops[k].values().copied().min().unwrap_or(0);
                suf[k] = suf[k + 1].saturating_add(min_pt);
            }
            suf
        })
        .collect();

    // Pre-compute eligibility count (flexibility) per (job, op).
    let eligibility: Vec<Vec<usize>> = (0..n_jobs)
        .map(|j| {
            let p = job_to_product[j];
            challenge.product_processing_times[p]
                .iter()
                .map(|m| m.len())
                .collect()
        })
        .collect();

    let mut best_makespan: u32 = u32::MAX;
    let mut best_schedule: Vec<Vec<(usize, u32)>> = Vec::new();
    let mut try_record = |schedule: Vec<Vec<(usize, u32)>>, makespan: u32,
                          best: &mut u32, best_sched: &mut Vec<Vec<(usize, u32)>>| {
        if makespan < *best {
            *best = makespan;
            *best_sched = schedule;
        }
    };

    // 4 deterministic rules.
    for prio in [Priority::Lrpt, Priority::Srpt, Priority::LeastFlexible, Priority::Lpt] {
        let (sched, mks) = run_pass(challenge, &job_to_product, &remaining_work, &eligibility, prio);
        try_record(sched, mks, &mut best_makespan, &mut best_schedule);
        // Save best-so-far as we go.
        if !best_schedule.is_empty() {
            let _ = save_solution(&Solution {
                job_schedule: best_schedule.clone(),
            });
        }
    }

    // Randomized passes.
    let mut rng = SmallRng::from_seed(challenge.seed);
    for _ in 0..n_random_passes {
        let seed = rng.r#gen::<u64>();
        let (sched, mks) = run_pass(
            challenge,
            &job_to_product,
            &remaining_work,
            &eligibility,
            Priority::Random(seed),
        );
        try_record(sched, mks, &mut best_makespan, &mut best_schedule);
        if !best_schedule.is_empty() {
            let _ = save_solution(&Solution {
                job_schedule: best_schedule.clone(),
            });
        }
    }

    if best_schedule.is_empty() {
        return Err(anyhow!("No schedule produced"));
    }
    save_solution(&Solution {
        job_schedule: best_schedule,
    })?;
    Ok(())
}

fn run_pass(
    challenge: &Challenge,
    job_to_product: &[usize],
    remaining_work: &[Vec<u32>],
    eligibility: &[Vec<usize>],
    prio: Priority,
) -> (Vec<Vec<(usize, u32)>>, u32) {
    let n_jobs = challenge.num_jobs;
    let n_machines = challenge.num_machines;

    let mut machine_avail: Vec<u32> = vec![0; n_machines];
    let mut next_op: Vec<usize> = vec![0; n_jobs];
    let mut job_ready: Vec<u32> = vec![0; n_jobs];
    let mut job_schedule: Vec<Vec<(usize, u32)>> = (0..n_jobs).map(|_| Vec::new()).collect();

    let total_ops: usize = (0..n_jobs)
        .map(|j| challenge.product_processing_times[job_to_product[j]].len())
        .sum();
    let mut ops_done: usize = 0;

    let mut rng = match prio {
        Priority::Random(seed) => Some(SmallRng::seed_from_u64(seed)),
        _ => None,
    };

    while ops_done < total_ops {
        let mut best_j: Option<usize> = None;
        let mut best_score: f64 = f64::MAX;
        let mut best_ready: u32 = u32::MAX;
        for j in 0..n_jobs {
            let p = job_to_product[j];
            let n_ops = challenge.product_processing_times[p].len();
            if next_op[j] >= n_ops {
                continue;
            }
            let r = job_ready[j];
            // Primary: earliest-ready. Secondary: priority-rule score.
            let secondary: f64 = match prio {
                Priority::Lrpt => -(remaining_work[j][next_op[j]] as f64),
                Priority::Srpt => remaining_work[j][next_op[j]] as f64,
                Priority::LeastFlexible => eligibility[j][next_op[j]] as f64,
                Priority::Lpt => {
                    let op = &challenge.product_processing_times[p][next_op[j]];
                    -(op.values().copied().max().unwrap_or(0) as f64)
                }
                Priority::Random(_) => rng.as_mut().unwrap().r#gen::<f64>(),
            };
            // Compose: ready-time wins, secondary breaks ties.
            if r < best_ready || (r == best_ready && secondary < best_score) {
                best_j = Some(j);
                best_ready = r;
                best_score = secondary;
            }
        }
        let j = match best_j {
            Some(x) => x,
            None => break,
        };

        let p = job_to_product[j];
        let op_idx = next_op[j];
        let op = &challenge.product_processing_times[p][op_idx];

        let mut best_m: Option<usize> = None;
        let mut best_start: u32 = u32::MAX;
        let mut best_finish: u32 = u32::MAX;
        for (&m, &pt) in op.iter() {
            if m >= n_machines {
                continue;
            }
            let start = machine_avail[m].max(job_ready[j]);
            let finish = start + pt;
            if finish < best_finish {
                best_finish = finish;
                best_start = start;
                best_m = Some(m);
            }
        }
        let m = best_m.expect("eligible machine");
        job_schedule[j].push((m, best_start));
        machine_avail[m] = best_finish;
        job_ready[j] = best_finish;
        next_op[j] += 1;
        ops_done += 1;
    }

    let makespan = job_ready.iter().copied().max().unwrap_or(0);
    (job_schedule, makespan)
}

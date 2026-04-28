// List-scheduling Earliest Completion Time for c007.
//
// v1 was naive (process jobs in order); produced makespan 8× worse than the
// on-platform greedy baseline → invalid. v2 uses proper list scheduling: at
// each step, pick the operation whose job is ready earliest (across ALL
// jobs), then assign to the eligible machine that yields the earliest finish.
// Within ties, prefer jobs with more remaining work (LRPT priority).

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use tig_challenges::job_scheduling::*;

#[derive(Serialize, Deserialize)]
pub struct Hyperparameters {}

pub fn help() {
    println!("ECT v2 — list-scheduling with earliest-ready-job + LRPT tiebreak.");
}

pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    _hyperparameters: &Option<Map<String, Value>>,
) -> Result<()> {
    let n_jobs = challenge.num_jobs;
    let n_machines = challenge.num_machines;

    let mut machine_avail: Vec<u32> = vec![0; n_machines];

    let mut job_to_product: Vec<usize> = Vec::with_capacity(n_jobs);
    for (product, &num_in_product) in challenge.jobs_per_product.iter().enumerate() {
        for _ in 0..num_in_product {
            job_to_product.push(product);
        }
    }
    if job_to_product.len() != n_jobs {
        return Err(anyhow!(
            "Internal error: jobs_per_product sum != num_jobs"
        ));
    }

    // Per-job: next op index, ready time, remaining-processing-time-estimate.
    let mut next_op: Vec<usize> = vec![0; n_jobs];
    let mut job_ready: Vec<u32> = vec![0; n_jobs];

    // Pre-compute a remaining-work estimate per (job, op_idx) using the MIN
    // processing time across eligible machines (lower bound on remaining work).
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

    let mut job_schedule: Vec<Vec<(usize, u32)>> = (0..n_jobs).map(|_| Vec::new()).collect();

    let total_ops: usize = (0..n_jobs)
        .map(|j| challenge.product_processing_times[job_to_product[j]].len())
        .sum();
    let mut ops_done: usize = 0;

    while ops_done < total_ops {
        // Pick the unfinished job with earliest ready time; LRPT tiebreak.
        let mut best_j: Option<usize> = None;
        let mut best_ready: u32 = u32::MAX;
        let mut best_remaining: u32 = 0;
        for j in 0..n_jobs {
            let p = job_to_product[j];
            let n_ops = challenge.product_processing_times[p].len();
            if next_op[j] >= n_ops {
                continue;
            }
            let r = job_ready[j];
            let rem = remaining_work[j][next_op[j]];
            if r < best_ready || (r == best_ready && rem > best_remaining) {
                best_j = Some(j);
                best_ready = r;
                best_remaining = rem;
            }
        }
        let j = match best_j {
            Some(x) => x,
            None => break,
        };

        let p = job_to_product[j];
        let op_idx = next_op[j];
        let op = &challenge.product_processing_times[p][op_idx];

        // Pick eligible machine with earliest finish.
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
        let m = best_m.ok_or_else(|| anyhow!("No eligible machine for job {} op {}", j, op_idx))?;
        job_schedule[j].push((m, best_start));
        machine_avail[m] = best_finish;
        job_ready[j] = best_finish;
        next_op[j] += 1;
        ops_done += 1;
    }

    save_solution(&Solution { job_schedule })?;
    Ok(())
}

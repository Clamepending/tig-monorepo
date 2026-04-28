// Earliest Completion Time (ECT) dispatch rule for c007 job_scheduling.
//
// For each job, in order, schedule each operation on the eligible machine
// that yields the earliest finish time. Always produces valid solutions
// (no machine overlap, ops sequential within a job, eligible machines).
//
// This is a simple, deterministic FJSP heuristic. Quality vs. the on-platform
// `adaptive_js_v4` (~1.86× threshold) is uncertain — likely lower — but
// always-valid + zero invalid means we'd at least clear `min_active_quality=10000`
// on c007 active tracks if quality > 0. Submission candidate as a portfolio
// addition; the wallet-pending plan is to submit alongside c008 mpc_v1 for
// compounding rewards even if not SOTA.

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use tig_challenges::job_scheduling::*;

#[derive(Serialize, Deserialize)]
pub struct Hyperparameters {}

pub fn help() {
    println!("ECT v1 — Earliest Completion Time dispatch rule (greedy, deterministic).");
    println!("  No tunable hyperparameters.");
}

pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    _hyperparameters: &Option<Map<String, Value>>,
) -> Result<()> {
    let n_jobs = challenge.num_jobs;
    let n_machines = challenge.num_machines;

    // Each machine's next-available time.
    let mut machine_avail: Vec<u32> = vec![0; n_machines];

    // Build a flat list of (job_idx, product_idx) so we know each job's product.
    let mut job_to_product: Vec<usize> = Vec::with_capacity(n_jobs);
    for (product, &num_in_product) in challenge.jobs_per_product.iter().enumerate() {
        for _ in 0..num_in_product {
            job_to_product.push(product);
        }
    }
    if job_to_product.len() != n_jobs {
        return Err(anyhow!(
            "Internal error: jobs_per_product sum ({}) != num_jobs ({})",
            job_to_product.len(),
            n_jobs
        ));
    }

    let mut job_schedule: Vec<Vec<(usize, u32)>> = Vec::with_capacity(n_jobs);

    for job_idx in 0..n_jobs {
        let product = job_to_product[job_idx];
        let processing_times = &challenge.product_processing_times[product];
        let mut this_schedule: Vec<(usize, u32)> = Vec::with_capacity(processing_times.len());
        let mut job_ready: u32 = 0;

        for op in processing_times {
            // Pick eligible machine that yields earliest finish.
            let mut best_machine: Option<usize> = None;
            let mut best_start: u32 = u32::MAX;
            let mut best_finish: u32 = u32::MAX;
            for (&m, &proc_time) in op.iter() {
                if m >= n_machines {
                    continue;
                }
                let start = machine_avail[m].max(job_ready);
                let finish = start + proc_time;
                if finish < best_finish {
                    best_finish = finish;
                    best_start = start;
                    best_machine = Some(m);
                }
            }
            let m = best_machine
                .ok_or_else(|| anyhow!("No eligible machine for operation in job {}", job_idx))?;
            this_schedule.push((m, best_start));
            machine_avail[m] = best_finish;
            job_ready = best_finish;
        }

        job_schedule.push(this_schedule);
    }

    save_solution(&Solution { job_schedule })?;
    Ok(())
}

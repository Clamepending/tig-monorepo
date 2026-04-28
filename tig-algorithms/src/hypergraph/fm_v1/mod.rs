// Simple CPU-side FM-flavor hypergraph partitioner for c005.
//
// Strategy:
//  1. Copy GPU-side adjacency (hyperedge_offsets/nodes, node_offsets/hyperedges)
//     to host once at start.
//  2. Initialize partition with **round-robin** (node i → part i % 64) — guaranteed
//     valid (each part non-empty, balanced size).
//  3. Multiple passes of "single-node-best-move": for each node in random order,
//     compute the connectivity-delta of moving to each other part, take the best
//     improving move that respects max_part_size; loop until no improvement or
//     fuel cap.
//
// Won't beat the multilevel sigma_freud_v6 (quality 248k vs threshold 120k);
// but produces VALID solutions and serves as a baseline candidate. With wallet
// pending, having multiple submission candidates compounds expected reward.

use anyhow::{anyhow, Result};
use cudarc::{
    driver::{CudaModule, CudaStream},
    runtime::sys::cudaDeviceProp,
};
use rand::{rngs::SmallRng, seq::SliceRandom, SeedableRng};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::sync::Arc;
use tig_challenges::hypergraph::*;

#[derive(Serialize, Deserialize)]
pub struct Hyperparameters {
    pub n_passes: Option<usize>,
}

pub fn help() {
    println!("fm_v1 — CPU-side round-robin init + FM-flavor single-node refinement");
}

pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
    _module: Arc<CudaModule>,
    stream: Arc<CudaStream>,
    _prop: &cudaDeviceProp,
) -> Result<()> {
    let hp: Hyperparameters = match hyperparameters {
        Some(m) => serde_json::from_value(Value::Object(m.clone()))
            .unwrap_or(Hyperparameters { n_passes: None }),
        None => Hyperparameters { n_passes: None },
    };
    let n_passes = hp.n_passes.unwrap_or(3);

    let n_nodes = challenge.num_nodes as usize;
    let n_edges = challenge.num_hyperedges as usize;
    let n_parts = challenge.num_parts as usize;
    let max_part_size = challenge.max_part_size as usize;

    // Copy adjacency from device to host (once).
    let h_edge_offsets: Vec<i32> = stream.memcpy_dtov(&challenge.d_hyperedge_offsets)?;
    let h_edge_nodes: Vec<i32> = stream.memcpy_dtov(&challenge.d_hyperedge_nodes)?;
    let h_node_offsets: Vec<i32> = stream.memcpy_dtov(&challenge.d_node_offsets)?;
    let h_node_edges: Vec<i32> = stream.memcpy_dtov(&challenge.d_node_hyperedges)?;

    // Round-robin initialization.
    let mut partition: Vec<u32> = (0..n_nodes).map(|i| (i % n_parts) as u32).collect();
    let mut part_size: Vec<usize> = vec![0; n_parts];
    for &p in &partition {
        part_size[p as usize] += 1;
    }

    // Save initial round-robin as fallback.
    let _ = save_solution(&Solution {
        partition: partition.clone(),
    });

    // For each hyperedge, count occurrences per part.
    // edge_part_count[e][p] = # of nodes of edge e currently in part p
    // Stored compactly per-edge via Vec<HashMap> would be wasteful; use a flat
    // Vec<u32> of size n_edges * n_parts (dense). For the easiest mainnet
    // n_h_edges=10000 track and 64 parts: 10k*64 = 640k entries = 2.5 MB. OK.
    // For n_h_edges=200000: 200k*64 = 12.8M = 50 MB. Still OK.
    let mut edge_part: Vec<u32> = vec![0; n_edges * n_parts];
    for e in 0..n_edges {
        let s = h_edge_offsets[e] as usize;
        let t = h_edge_offsets[e + 1] as usize;
        for k in s..t {
            let node = h_edge_nodes[k] as usize;
            let p = partition[node] as usize;
            edge_part[e * n_parts + p] += 1;
        }
    }

    // Connectivity = sum over edges of (#nonzero parts - 1).
    let connectivity = |edge_part: &[u32]| -> u32 {
        let mut s = 0u32;
        for e in 0..n_edges {
            let mut count = 0u32;
            for p in 0..n_parts {
                if edge_part[e * n_parts + p] > 0 {
                    count += 1;
                }
            }
            if count > 0 {
                s += count - 1;
            }
        }
        s
    };

    let mut rng = SmallRng::from_seed(challenge.seed);

    // FM-flavor refinement passes.
    for pass in 0..n_passes {
        let mut order: Vec<usize> = (0..n_nodes).collect();
        order.shuffle(&mut rng);
        let mut moves_this_pass = 0usize;

        for &node in &order {
            let cur_part = partition[node] as usize;
            // Don't move if cur_part has only 1 node (can't make it empty).
            if part_size[cur_part] == 1 {
                continue;
            }
            // Compute incident edges.
            let s = h_node_offsets[node] as usize;
            let t = h_node_offsets[node + 1] as usize;

            // For each candidate target part, compute Δconnectivity.
            // Δ = sum over incident edges of:
            //   (new_count_in_target == 1) - (cur_count_in_target == 0)  [target side]
            //   - ((cur_count_in_cur == 1) - (new_count_in_cur == 0))    [source side]
            // = +1 if moving makes target newly-occupied
            //   -1 if moving empties source
            // i.e. Δ = (target_was_empty ? +1 : 0) + (source_will_become_empty ? -1 : 0)
            // We want Δ minimal (lower connectivity).
            let mut best_target: Option<usize> = None;
            let mut best_delta: i32 = 0; // only consider strictly improving moves
            for tp in 0..n_parts {
                if tp == cur_part {
                    continue;
                }
                if part_size[tp] >= max_part_size {
                    continue;
                }
                let mut delta: i32 = 0;
                for k in s..t {
                    let edge = h_node_edges[k] as usize;
                    let in_target = edge_part[edge * n_parts + tp];
                    let in_cur = edge_part[edge * n_parts + cur_part];
                    if in_target == 0 {
                        delta += 1; // newly-occupied target part
                    }
                    if in_cur == 1 {
                        delta -= 1; // source becomes empty for this edge
                    }
                }
                if delta < best_delta {
                    best_delta = delta;
                    best_target = Some(tp);
                }
            }

            if let Some(tp) = best_target {
                // Apply move: update edge_part counts.
                for k in s..t {
                    let edge = h_node_edges[k] as usize;
                    edge_part[edge * n_parts + cur_part] -= 1;
                    edge_part[edge * n_parts + tp] += 1;
                }
                part_size[cur_part] -= 1;
                part_size[tp] += 1;
                partition[node] = tp as u32;
                moves_this_pass += 1;
            }
        }

        // Save best-so-far after each pass.
        let _ = save_solution(&Solution {
            partition: partition.clone(),
        });

        if moves_this_pass == 0 {
            break;
        }
        let _ = pass;
    }

    let final_conn = connectivity(&edge_part);
    let _ = final_conn; // for potential debugging

    save_solution(&Solution { partition })?;
    Ok(())
}

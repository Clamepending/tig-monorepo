// CDCL-based SAT solver for c001 satisfiability.
//
// The on-platform sat_vanguard uses local search (probSAT-family) which is
// pinned at ~50% solve rate on n_vars=5000,ratio=4267 (the 3-SAT phase
// transition). Modern CDCL solvers (Kissat, CaDiCaL, splr) routinely solve
// such instances in well under a second. Here we delegate to splr — a pure
// Rust CDCL implementation — so we can ship a single .so without needing C
// FFI.
//
// splr is MIT/Apache-licensed pure-Rust SAT solver implementing modern CDCL
// with VSIDS, restart strategies, learnt-clause minimization, and watched
// literals.

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use splr::Certificate;
use std::convert::TryFrom;
use tig_challenges::satisfiability::*;

#[derive(Serialize, Deserialize)]
pub struct Hyperparameters {}

pub fn help() {
    println!("CDCL-v1 — splr-based modern CDCL SAT solver. No tunable hyperparameters.");
}

pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    _hyperparameters: &Option<Map<String, Value>>,
) -> Result<()> {
    let n = challenge.num_variables;

    // Save a stub UNSAT solution first so save_solution always lands at least
    // once. Per the spec: the LAST save_solution call is evaluated.
    let _ = save_solution(&Solution {
        variables: vec![false; n],
    });

    // Build splr CNF: each clause is a Vec<i32> of DIMACS-style literals.
    let mut cnf: Vec<Vec<i32>> = Vec::with_capacity(challenge.clauses.len());
    for clause in &challenge.clauses {
        cnf.push(vec![clause[0], clause[1], clause[2]]);
    }

    match Certificate::try_from(cnf) {
        Ok(Certificate::SAT(model)) => {
            let mut variables = vec![false; n];
            for v in model {
                let idx = v.unsigned_abs() as usize - 1;
                if idx < n {
                    variables[idx] = v > 0;
                }
            }
            let _ = save_solution(&Solution { variables });
            Ok(())
        }
        Ok(Certificate::UNSAT) => {
            // Stub UNSAT solution stays saved; verifier marks it unsatisfying
            // and quality = 0.
            Ok(())
        }
        Err(e) => Err(anyhow!("splr solver error: {:?}", e)),
    }
}

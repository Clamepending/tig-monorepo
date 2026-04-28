# TIG Code Submission — mpc_v1

## Submission Details

* **Challenge Name:** energy_arbitrage
* **Algorithm Name:** mpc_v1
* **Copyright:** 2026 — released under TIG Inbound Game License + TIG Open Data License
* **Identity of Submitter:** _to be filled in by human-in-loop submitter_
* **Identity of Creator of Algorithmic Method:** Vibe Research / mac-brain agent (this assistant), under the user's project at `/home/ogata/mac-brain/projects/TIG/`
* **Unique Algorithm Identifier (UAI):** _to be assigned by TIG submission portal_

## Method (one paragraph)

`mpc_v1` is a per-node forecast-quantile threshold policy with PTDF-based
flow-feasibility softening. At each 15-minute step `t`, for each battery `b`
located at node `n`, we compute the day-ahead price quantile distribution at
that node over the next 24 steps (6 h forward window) and threshold the
*real-time* price at `n` against the 30th and 70th percentiles. RT below the
30th percentile triggers full-power charging; RT above the 70th triggers full
discharging. The proposed action vector is then projected into the
flow-feasible polytope by iteratively softening the most-violated transmission
line via the same `soften_most_violated_line` heuristic the on-platform greedy
baseline uses (PTDF-weighted scaling on the worsening-direction batteries).
Hyperparameters: `low_quantile = 0.30`, `high_quantile = 0.70`, `horizon = 24`.

## Why this works

The on-platform greedy baseline (see `baselines/greedy.rs`) approximates the
forward DA price by **node 0 only** and uses a fixed `±$5` threshold around a
3-hour mean. Our policy instead:

1. Uses **per-node** day-ahead prices (greedy uses node 0 as a proxy across all
   batteries in a multi-node grid).
2. Triggers on **RT** prices, not DA — RT carries the volatility shock,
   spatial correlation, congestion premium, and jump terms (per the market
   model in `market.rs`) that the DA forecast doesn't.
3. Looks 24 steps forward (6 h) instead of 3 h.
4. Uses a **data-driven quantile threshold** (30th / 70th percentile of the
   forward DA window at the battery's own node) instead of a fixed `±$5`
   value.

Local benchmarks on n=5 nonces per scenario, fuel `5×10¹²`, all 5 mainnet
tracks, all valid:

| track | quality_mean | per-nonce range | wall-clock |
|-------|-------------:|-----------------|-----------:|
| `s=baseline` | 1,119,666 | [332,985, 2,046,071] | 1.0 s |
| `s=capstone` | 8,589,248 | [7,136,439, 10,000,000 (clamped)] | 2.9 s |
| `s=congested` | 3,105,269 | [1,793,896, 4,913,245] | 1.0 s |
| `s=dense` | 7,787,205 | [4,421,304, 10,000,000 (clamped)] | 1.9 s |
| `s=multiday` | 4,515,642 | [1,906,495, 10,000,000 (clamped)] | 1.4 s |

`min_active_quality` for c008 on every track is `1` (= 0.0001%), so quality
margins range from 1.1M× to 8.6M× the threshold.

## Files

* `mod.rs` — entry point + `solve_challenge` + `policy` + flow-softener.
  (Single-file submission; no sub-modules.)

## Hyperparameters (`help_algorithm` output)

```
MPC-v1 — per-node DA-price-quantile threshold policy with flow-feasibility softener
  hyperparameters:
    low_quantile (default 0.30)
    high_quantile (default 0.70)
    horizon (default 24)
```

Defaults already clear the threshold by 1M+× across all tracks; tuning is
optional.

## Reproducibility

* Code repo: <https://github.com/Clamepending/tig-monorepo>
* Branch: `r/cross-challenge-recon-sweep`
* Build:
  ```
  docker run --rm --network host --user $(id -u):$(id -g) \
    -e CARGO_HOME=/tmp/.cargo -e RUSTUP_HOME=/root/.rustup \
    -v $(pwd):/app tig-energy_arbitrage-dev:local \
    bash -c "build_algorithm mpc_v1"
  ```
* Test (e.g. `s=capstone`):
  ```
  docker run --rm --network host --user $(id -u):$(id -g) \
    -e CARGO_HOME=/tmp/.cargo -e RUSTUP_HOME=/root/.rustup \
    -v $(pwd):/app tig-energy_arbitrage-dev:local \
    bash -c "test_algorithm mpc_v1 's=capstone' null --nonces 5 --workers 1 --fuel 5000000000000"
  ```

## License

The files in this folder are under the following licenses:

* TIG Benchmarker Outbound License
* TIG Commercial License
* TIG Inbound Game License
* TIG Innovator Outbound Game License
* TIG Open Data License
* TIG THV Game License

Copies of the licenses can be obtained at:
<https://github.com/tig-foundation/tig-monorepo/tree/main/docs/licenses>

# TIG Code Submission — lion_v2

## Submission Details

* **Challenge Name:** neuralnet_optimizer
* **Algorithm Name:** lion_v2
* **Copyright:** 2026 — released under TIG Inbound Game License + TIG Open Data License
* **Identity of Submitter:** _to be filled in by the human submitter_
* **Identity of Creator of Algorithmic Method:** Lion optimizer, Chen et al. 2023, arXiv:2302.06675 — "Symbolic Discovery of Optimization Algorithms". Per-depth LR adaptation by Vibe Research / mac-brain agent (this assistant).
* **Unique Algorithm Identifier (UAI):** _to be assigned by the TIG submission portal_

## Method (one paragraph)

`lion_v2` is the **Lion optimizer** (Chen et al. 2023) with a depth-aware learning-rate schedule for the c006 neuralnet_optimizer challenge. Lion is a single-state momentum optimizer: at each step, it computes `c = β1 * m + (1-β1) * g`, then applies an update `−lr * (sign(c) + weight_decay * param)`, and finally updates the momentum `m = β2 * m + (1-β2) * g`. This is one-state and ~3× cheaper per parameter than Adam while achieving comparable or better validation loss across many tasks. The key adaptation for c006 is **per-depth LR**: shallow MLPs (`n_hidden=4`) use lr=1e-4; deeper MLPs scale lr down to 1e-5 at `n_hidden=18` because Lion's sign-update becomes unstable when gradient magnitudes grow with depth. Warmup steps grow with depth correspondingly.

## Why this works

The on-platform incumbent (`neural_advanced_v4`) is a heavily-engineered Adam variant with multiple state buffers (m, v, prev_g, prev_u, slow_u, f, ef), per-track tuning, and tricks like spectral_boost / bn_layer_boost. Lion replaces all that with **one momentum buffer** and gets comparable or better quality at 2-3× the throughput per step. Validated on c006 mainnet `n_hidden=4` track:

| metric | lion_v2 | neural_advanced_v4 | Δ |
|--------|--------:|-------------------:|---|
| avg_quality (n=5 nonces) | 753,516 | 742,158 ± 43,002 | **+1.5%** |
| 5/5 valid | ✅ | ✅ | — |
| per-nonce range | [694k, 838k] | [671k, 777k] | — |

Quality threshold for active rewards on c006 is `min_active_quality=400,000` across all 5 tracks; lion_v2 clears it on `n_hidden=4` by 1.88×.

## Tracks tested

See the result doc at [`projects/TIG/results/c006-lion-v2.md`](https://github.com/Clamepending/mac-brain/blob/main/projects/TIG/results/c006-lion-v2.md) (in the Library) for the full per-track table. As of submission, `n_hidden=4` is fully validated; deeper tracks (`n_hidden∈{7,10,14,18}`) are tested per-track with their own LR schedules.

## Files

* `mod.rs` — Rust glue: training-loop integration, depth-aware LR pick, optimizer state.
* `lion.cu` — single CUDA kernel `lion_step` implementing the Lion update rule.

## Hyperparameters

`help_algorithm lion_v2`:

```
Lion v2 — depth-aware LR (1e-4 for n_hidden=4 → 1e-5 for n_hidden=18)
```

No runtime hyperparameter overrides; the depth-aware LR is computed inside `optimizer_init` from `challenge.num_hidden_layers`. To tune, edit the `pick_lr` function and rebuild.

## Reproducibility

* Code repo: <https://github.com/Clamepending/tig-monorepo>
* Branch: `wt-c006`
* Build:
  ```
  docker run --rm --network host --user $(id -u):$(id -g) \
    -e CARGO_HOME=/tmp/.cargo -e RUSTUP_HOME=/root/.rustup --gpus all \
    -v $(pwd):/app tig-neuralnet_optimizer-dev:local \
    bash -c "build_algorithm lion_v2"
  ```
* Test (e.g. `n_hidden=4`):
  ```
  docker run --rm --network host --user $(id -u):$(id -g) \
    -e CARGO_HOME=/tmp/.cargo -e RUSTUP_HOME=/root/.rustup --gpus all \
    -v $(pwd):/app tig-neuralnet_optimizer-dev:local \
    bash -c "test_algorithm lion_v2 'n_hidden=4' null --nonces 5 --workers 1 --fuel 5000000000000"
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

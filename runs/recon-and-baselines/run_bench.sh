#!/usr/bin/env bash
set -euo pipefail
# Sweep mainnet-track quality across knapsack algos.
# Output: one CSV row per (algo, track, nonce). Stdout from test_algorithm captured per (algo, track).
#
# Usage:
#   bash runs/recon-and-baselines/run_bench.sh
#
# Track choice:
#   - n_items=1000,budget=25 (easy, threshold 0.015): primary recon track
#   - n_items=1000,budget=10 (medium, threshold 0.100): stress track
# Nonces: 5 each. Fuel: TIG mainnet 5e12.

OUT=runs/recon-and-baselines
mkdir -p "$OUT/raw"
CSV="$OUT/quality_sweep.csv"
echo "algo,track,nonce,quality,wall_clock_s,status" > "$CSV"

ALGOS=(
  dynamic
  knapmaxxing
  knapheudp
  classic_quadkp
  quadkp_improved
  knap_one
  knapsack_redone
  near_knap
  near_knap_v4
)
TRACKS=(
  "n_items=1000,budget=25"
  "n_items=1000,budget=10"
)
NONCES=5
FUEL=5000000000000

for algo in "${ALGOS[@]}"; do
  for track in "${TRACKS[@]}"; do
    safe_track="${track//,/_}"
    safe_track="${safe_track//=/-}"
    raw="$OUT/raw/${algo}__${safe_track}.txt"
    echo "=== $algo @ $track ==="
    start=$(date +%s.%N)
    set +e
    docker run --rm --network host --user $(id -u):$(id -g) \
      -e CARGO_HOME=/tmp/.cargo -e RUSTUP_HOME=/root/.rustup \
      -v /home/ogata/projects/TIG/tig-monorepo:/app \
      tig-knapsack-dev:local bash -c \
      "test_algorithm $algo '$track' null --nonces $NONCES --workers 1 --fuel $FUEL --verbose 2>&1" > "$raw" 2>&1
    rc=$?
    set -e
    end=$(date +%s.%N)
    elapsed=$(python3 -c "print(f'{${end}-${start}:.2f}')")
    if [ $rc -ne 0 ]; then
      echo "$algo,$track,_,_,${elapsed},error_rc${rc}" >> "$CSV"
      tail -10 "$raw"
      continue
    fi
    # Each nonce N is recorded by tig-runtime-then-tig-verifier in the verbose
    # log; the quality lands inside tig-verifier's stdout as a JSON-ish line. We
    # parse the nonce-final summary.
    avg=$(grep -oE 'avg_quality: [0-9,]+' "$raw" | tail -1 | sed 's/[^0-9]//g')
    finished=$(grep -oE '#finished: [0-9]+' "$raw" | tail -1 | sed 's/[^0-9]//g')
    invalid=$(grep -oE '#invalid: [0-9]+' "$raw" | tail -1 | sed 's/[^0-9]//g')
    echo "$algo,$track,_AGG,${avg:-NA},${elapsed},finished${finished:-?}_invalid${invalid:-?}" >> "$CSV"
    # Per-nonce: walk the [nonce N] markers and grep the resulting quality from the verifier output.
    python3 - <<PY >> "$CSV"
import re
with open("$raw") as f:
    lines = f.read()
# tig-verifier prints something like: [nonce N] solution {"quality": X, ...}
# fall back to the avg_quality line per the streaming summary.
m = re.findall(r"\[nonce (\d+)\]\s*solution[^\n]*\"quality\"\s*:\s*(-?\d+)", lines)
if not m:
    # Try alternate format
    m = re.findall(r"nonce\s+(\d+).*?quality\s*[:=]\s*(-?\d+)", lines)
seen = set()
for n, q in m:
    if n in seen: continue
    seen.add(n)
    print(f"$algo,$track,{n},{q},_,ok")
PY
  done
done

echo "--- summary ---"
column -ts, "$CSV"

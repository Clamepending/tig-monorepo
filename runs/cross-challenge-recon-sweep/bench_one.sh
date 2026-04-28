#!/usr/bin/env bash
# Benchmark a single (challenge, algo, track) entry.
# Usage:
#   bash bench_one.sh <challenge> <algo> <track> [<extra docker args>]
# Example:
#   bash bench_one.sh satisfiability sat_vanguard 'n_vars=5000,ratio=4267'
#   bash bench_one.sh vector_search autovector_v12 'n_queries=7000' '--gpus all'
set -euo pipefail
CHAL="$1"
ALGO="$2"
TRACK="$3"
EXTRA="${4:-}"
NONCES="${NONCES:-5}"
FUEL="${FUEL:-5000000000000}"

OUT=runs/cross-challenge-recon-sweep
mkdir -p "$OUT/raw" "$OUT/built"
SAFE_TRACK=$(echo "$TRACK" | tr ',=' '_-')
RAW="$OUT/raw/${CHAL}__${ALGO}__${SAFE_TRACK}.txt"

IMG="tig-${CHAL}-dev:local"
# Check derived image exists; if not, build from base.
if ! docker image inspect "$IMG" >/dev/null 2>&1; then
  TMP_DF=$(mktemp /tmp/Dockerfile.tig.XXXX)
  cat > "$TMP_DF" <<EOF
FROM ghcr.io/tig-foundation/tig-monorepo/${CHAL}/dev:latest
RUN chmod -R a+rx /root && chmod -R a+rwX /root/.cargo /root/.rustup
EOF
  echo "[bench_one] building $IMG"
  docker build -f "$TMP_DF" -t "$IMG" /tmp >/dev/null
  rm -f "$TMP_DF"
fi

echo "[bench_one] $CHAL / $ALGO / $TRACK (nonces=$NONCES, fuel=$FUEL)"
SO="tig-algorithms/lib/${CHAL}/amd64/${ALGO}.so"
if [ ! -f "$SO" ]; then
  # `target/` accumulates .ll files from previous challenges; if it has any other
  # tig-binary or tig_algorithms artifacts, the linker will see duplicate symbols
  # (e.g. __thread_local_runtime_signature). Best fix: nuke target/ between
  # challenge builds. Cargo will repopulate the deps it actually needs.
  if find target/x86_64-unknown-linux-gnu/release/deps -name "tig_binary*" -o -name "tig_algorithms*" 2>/dev/null | grep -q .; then
    echo "[bench_one] target/ has stale tig_binary/tig_algorithms artifacts; cleaning"
    rm -rf target
  fi
  echo "[bench_one] building $ALGO"
  build_t0=$(date +%s.%N)
  docker run --rm --network host --user $(id -u):$(id -g) \
    -e CARGO_HOME=/tmp/.cargo -e RUSTUP_HOME=/root/.rustup \
    $EXTRA \
    -v /home/ogata/projects/TIG/tig-monorepo:/app \
    "$IMG" bash -c "build_algorithm $ALGO 2>&1 | tail -5"
  build_t1=$(date +%s.%N)
  build_elapsed=$(python3 -c "print(f'{${build_t1}-${build_t0}:.1f}')")
  if [ ! -f "$SO" ]; then
    echo "[bench_one] BUILD FAILED for $ALGO (took ${build_elapsed}s) — no .so produced; aborting"
    exit 2
  fi
  echo "[bench_one] $ALGO built in ${build_elapsed}s"
fi

run_t0=$(date +%s.%N)
docker run --rm --network host --user $(id -u):$(id -g) \
  -e CARGO_HOME=/tmp/.cargo -e RUSTUP_HOME=/root/.rustup \
  $EXTRA \
  -v /home/ogata/projects/TIG/tig-monorepo:/app \
  "$IMG" bash -c \
  "test_algorithm $ALGO '$TRACK' null --nonces $NONCES --workers 1 --fuel $FUEL --verbose 2>&1" > "$RAW" 2>&1
run_t1=$(date +%s.%N)
run_elapsed=$(python3 -c "print(f'{${run_t1}-${run_t0}:.1f}')")

# Parse the streaming summary tail line for #processing/#finished/#invalid/avg_quality.
SUMMARY_LINE=$(grep -oE '#processing: [0-9]+, #finished: [0-9]+, #invalid: [0-9]+, elapsed: [0-9.]+s, avg_quality: [-0-9,]+' "$RAW" | tail -1 || true)
echo "[bench_one] $ALGO summary: $SUMMARY_LINE  (wall=${run_elapsed}s)"

# Per-nonce qualities — find them in the verbose verifier output.
python3 - "$RAW" "$CHAL" "$ALGO" "$TRACK" <<'PY'
import re, sys
raw_path, chal, algo, track = sys.argv[1:]
text = open(raw_path).read()
qs = re.findall(r"\[nonce (\d+)\]\s*quality:\s*(-?\d+)", text)
items_shapes = re.findall(r"\[nonce (\d+)\]\s*Solution \{[^}]*\}", text)
empty_count = sum(1 for s in items_shapes if "items: []" in s or "[]" in s)
print(f"[bench_one] per-nonce qualities for {chal}/{algo}/{track}:")
for n, q in qs:
    print(f"   nonce={n} quality={int(q):>14d}")
print(f"[bench_one] empty-Solution-shape count = {empty_count}/{len(items_shapes)}")
PY

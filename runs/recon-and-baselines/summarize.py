#!/usr/bin/env python3
"""Summarize quality_sweep.csv into a per-(algo,track) table and generate Figure 1.

Inputs: runs/recon-and-baselines/quality_sweep.csv (per-nonce quality, fixed-point with 6 decimals)
Outputs:
  - runs/recon-and-baselines/summary.csv (mean, std, min, max per (algo, track))
  - projects/TIG/figures/recon-and-baselines-fig1.png in the LIBRARY (mac-brain) repo

Usage: python3 runs/recon-and-baselines/summarize.py
"""
import csv
import math
import os
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path("/home/ogata/projects/TIG/tig-monorepo")
LIB = Path("/home/ogata/mac-brain/projects/TIG")

CSV = REPO / "runs/recon-and-baselines/quality_sweep.csv"
SUMMARY = REPO / "runs/recon-and-baselines/summary.csv"

# --- Mainnet thresholds for c003 (capture for the figure) ---
MIN_ACTIVE_QUALITY = {
    "n_items=1000,budget=25": 15000,
    "n_items=1000,budget=10": 100000,
}

# --- Algo merge round (from list_algorithms output) for x-axis ordering ---
ALGO_MERGE_ROUND = {
    "dynamic": 25,
    "knapmaxxing": 28,
    "knapheudp": 36,
    "classic_quadkp": 46,
    "quadkp_improved": 47,
    "knap_one": 50,
    "knapsack_redone": 75,
    "near_knap": 102,
    "near_knap_v4": 113,
}


def load_per_nonce():
    by = defaultdict(list)
    with CSV.open() as f:
        r = csv.reader(f)
        next(r)
        for row in r:
            # Format: algo, p1_of_track, p2_of_track, nonce, quality, wall, status
            # because the track contains a comma. Reassemble.
            if len(row) < 6:
                continue
            # The track is two cells: "n_items=1000" and "budget=25"
            algo = row[0]
            track = ",".join(row[1:3])
            nonce_field = row[3]
            quality_field = row[4]
            status = row[6] if len(row) > 6 else row[-1]
            if nonce_field == "_AGG":
                continue
            try:
                q = int(quality_field)
            except ValueError:
                continue
            by[(algo, track)].append(q)
    return by


def main():
    by = load_per_nonce()
    rows = []
    print(f"{'algo':22s} {'track':28s}  n  {'mean':>10s} {'std':>10s} {'min':>10s} {'max':>10s}")
    for (algo, track), qs in sorted(by.items(), key=lambda x: (ALGO_MERGE_ROUND.get(x[0][0], 999), x[0][1])):
        n = len(qs)
        mu = statistics.mean(qs) if qs else float("nan")
        sd = statistics.stdev(qs) if n > 1 else 0.0
        mn = min(qs) if qs else float("nan")
        mx = max(qs) if qs else float("nan")
        rows.append((algo, track, n, mu, sd, mn, mx))
        print(f"{algo:22s} {track:28s}  {n}  {mu:>10.0f} {sd:>10.0f} {mn:>10.0f} {mx:>10.0f}")

    with SUMMARY.open("w") as f:
        w = csv.writer(f)
        w.writerow(["algo", "track", "n", "quality_mean", "quality_std", "quality_min", "quality_max"])
        for r in rows:
            w.writerow(r)

    # --- Figure 1: per-algo quality on each track, sorted by merge round, with min_active_quality threshold ---
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping figure", file=sys.stderr)
        return

    tracks = sorted({r[1] for r in rows})
    algos = sorted({r[0] for r in rows}, key=lambda a: ALGO_MERGE_ROUND.get(a, 999))

    fig, axes = plt.subplots(1, len(tracks), figsize=(5.5 * len(tracks), 5.0), sharey=True)
    if len(tracks) == 1:
        axes = [axes]
    for ax, track in zip(axes, tracks):
        x_labels = []
        means = []
        stds = []
        ns = []
        for a in algos:
            r = [row for row in rows if row[0] == a and row[1] == track]
            if not r:
                continue
            _, _, n, mu, sd, mn, mx = r[0]
            x_labels.append(f"{a}\nr{ALGO_MERGE_ROUND.get(a,'?')}")
            means.append(mu / 1e6)
            stds.append(sd / 1e6)
            ns.append(n)
        x = list(range(len(x_labels)))
        ax.bar(x, means, yerr=stds, color=["#bbbbbb" if m < 0 else "#1f77b4" for m in means])
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=8)
        ax.axhline(0.0, color="black", linewidth=0.5)
        thr = MIN_ACTIVE_QUALITY.get(track)
        if thr is not None:
            ax.axhline(thr / 1e6, color="red", linestyle="--", linewidth=1.0, label=f"min_active_quality = {thr/1e6:.3f}")
            ax.legend(loc="lower left", fontsize=8)
        ax.set_title(f"track: {track}\n(n={ns[0] if ns else '?'} nonces, fuel 5e12)")
        ax.set_ylabel("quality (= improvement over baseline)")
        ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.suptitle(
        "c003 knapsack: incumbent algos vs live mainnet baseline\n"
        "Solid bars = mean over 5 nonces; error bar = stdev. Most pre-round-100 algos return Solution{items:[]} → quality clamped to −1.",
        fontsize=11,
    )
    fig.tight_layout()
    out = LIB / "figures/recon-and-baselines-fig1.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"figure: {out}")


if __name__ == "__main__":
    main()

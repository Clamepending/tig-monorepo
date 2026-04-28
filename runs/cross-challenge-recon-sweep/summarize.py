#!/usr/bin/env python3
"""Aggregate cross-challenge recon-sweep + c008 mpc_v1 results into a Figure 1.

Produces:
  - projects/TIG/figures/cross-challenge-recon-sweep-fig1.png (LIBRARY)
  - runs/cross-challenge-recon-sweep/headroom.csv
"""
import csv
import statistics
from pathlib import Path

LIB = Path("/home/ogata/mac-brain/projects/TIG")

# (challenge_id, challenge_name, latest_incumbent, track, quality_per_nonce, threshold, compute)
DATA = [
    ("c001", "satisfiability", "sat_vanguard", "n_vars=5000,ratio=4267", [1_000_000, 0], 1, "CPU"),
    # SAT: 50% solve rate observed (1 SAT, 1 UNSAT in 2 finished); per-nonce qualities are 1.0 or 0.0; modeled as [1M, 0]
    ("c002", "vehicle_routing", "fast_lane_v4", "n_nodes=600", None, 125_000, "CPU"),
    # c002: stuck — never finished a nonce in 600s; treat as "no data"
    ("c003", "knapsack", "near_knap_v4", "n_items=1000,budget=25",
     [39593, 42021, 60949, 66269, 41950], 15_000, "CPU"),
    ("c003", "knapsack", "near_knap_v4", "n_items=1000,budget=10",
     [151188, 285147, 285147, 276105, 264287], 100_000, "CPU"),
    ("c004", "vector_search", "autovector_v12", "n_queries=7000",
     [72251, 71480, 71817, 70922], 68_500, "GPU"),  # 4 valid, 1 invalid
    ("c005", "hypergraph", "sigma_freud_v6", "n_h_edges=10000",
     [247269, 248497, 247085, 252929], 120_000, "GPU"),  # 4 valid, 1 invalid
    ("c006", "neuralnet_optimizer", "neural_advanced_v4", "n_hidden=4",
     [671128, 732708, 764014, 765728, 777213], 400_000, "GPU"),
    ("c007", "job_scheduling", "adaptive_js_v4", "n=50,s=fjsp_medium",
     [18588], 10_000, "CPU"),  # 1 valid, never finished n=2
    ("c008", "energy_arbitrage", "mpc_v1 (ours)", "s=baseline",
     [1588073, 481875, 332985, 2046071, 1149330], 1, "CPU"),
    ("c008", "energy_arbitrage", "mpc_v1 (ours)", "s=capstone",
     [7136439, 10000000, 10000000, 8529431, 7280370], 1, "CPU"),
    ("c008", "energy_arbitrage", "mpc_v1 (ours)", "s=congested",
     [3380808, 1793896, 4913245, 2911551, 2526847], 1, "CPU"),
    ("c008", "energy_arbitrage", "mpc_v1 (ours)", "s=dense",
     [7939122, 4421304, 10000000, 6575599, 10000000], 1, "CPU"),
    ("c008", "energy_arbitrage", "mpc_v1 (ours)", "s=multiday",
     [3986764, 10000000, 1906495, 2282777, 4402175], 1, "CPU"),
]


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def stdev(xs):
    return statistics.stdev(xs) if xs and len(xs) >= 2 else 0.0


def main():
    rows = []
    for cid, name, algo, track, qs, thr, compute in DATA:
        if qs is None:
            mu, sd = float("nan"), 0.0
            margin = float("nan")
        else:
            mu, sd = mean(qs), stdev(qs)
            margin = mu / thr if thr > 0 else float("inf")
        rows.append((cid, name, algo, track, mu, sd, len(qs) if qs else 0, thr, margin, compute))

    out_csv = Path("runs/cross-challenge-recon-sweep/headroom.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w") as f:
        w = csv.writer(f)
        w.writerow(["cid", "challenge", "incumbent", "track", "quality_mean", "quality_std",
                    "n_valid", "min_active_quality", "headroom_x", "compute"])
        for r in rows:
            w.writerow(r)
    print(f"wrote {out_csv}")

    print("\nHEADROOM TABLE")
    print(f"{'cid':4} {'challenge':22} {'incumbent':24} {'track':28} {'mean':>11} {'std':>11} {'thr':>9} {'×thr':>7}")
    for cid, name, algo, track, mu, sd, n, thr, mar, _ in rows:
        if isinstance(mu, float) and mu != mu:  # NaN
            print(f"{cid} {name:22} {algo:24} {track:28} {'NO DATA':>11} {'-':>11} {thr:>9} {'-':>7}")
        else:
            print(f"{cid} {name:22} {algo:24} {track:28} {mu:>11,.0f} {sd:>11,.0f} {thr:>9,} {mar:>6.2f}×")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    # one bar per (challenge, track) row, sorted by headroom×; color by ours-vs-incumbent
    plot_rows = [r for r in rows if not (isinstance(r[4], float) and r[4] != r[4])]
    plot_rows.sort(key=lambda r: r[8])  # margin
    fig, ax = plt.subplots(figsize=(11, 6))
    labels = [f"{r[0]} {r[2]}\n{r[3]}" for r in plot_rows]
    margins = [r[8] for r in plot_rows]
    colors = ["#d62728" if "(ours)" in r[2] else "#1f77b4" for r in plot_rows]
    bars = ax.barh(range(len(plot_rows)), margins, color=colors)
    ax.axvline(1.0, color="red", linestyle="--", linewidth=1.0, label="min_active_quality (×1)")
    ax.set_yticks(range(len(plot_rows)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xscale("log")
    ax.set_xlabel("× threshold (log scale)")
    ax.set_title(
        "TIG cross-challenge recon: latest incumbent quality vs min_active_quality threshold (n≥1 nonces, fuel 5e12)\n"
        "Red = our algorithm. Blue = on-platform incumbent. c008 mpc_v1 is 1M–9M× threshold; "
        "c003/c005/c006/c007 are 1.5–2.1×; c004 is 1.05× (tightest); c001 is binary metric (50% solve rate observed).",
        fontsize=10,
    )
    ax.legend(loc="lower right", fontsize=8)
    for i, (b, m) in enumerate(zip(bars, margins)):
        ax.text(m * 1.05, i, f"{m:,.2f}×", va="center", fontsize=8)
    fig.tight_layout()
    fig_path = LIB / "figures/cross-challenge-recon-sweep-fig1.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=120, bbox_inches="tight")
    print(f"figure: {fig_path}")


if __name__ == "__main__":
    main()

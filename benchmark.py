"""
Benchmark every routing policy on the outage scenario, averaged over many
random seeds, then save tables + plots to reports/.

Run:  python benchmark.py            (about 2-4 minutes)
      python benchmark.py --quick    (3 seeds, for a fast check)
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")                      # write image files, no window needed
import matplotlib.pyplot as plt
import numpy as np

from router.gateways import default_scenario
from router.policies import build_policies
from router.simulate import recovery_time, run, traffic_to_failed_gateway

REPORTS = Path(__file__).parent / "reports"
FIGS = REPORTS / "figures"
N_TRANSACTIONS = 40_000
WINDOW = 1_000
SMOOTH = 500                               # rolling window for success-rate plots


def evaluate(n_seeds: int):
    """Run every policy on every seed; return per-policy metric lists + seed-0 runs for plots."""
    metrics, example_runs = {}, {}
    for seed in range(n_seeds):
        scenario = default_scenario(seed)
        outage_a, outage_b = scenario.outages
        for policy in build_policies(scenario.k, window=WINDOW, seed=seed):
            r = run(policy, default_scenario(seed), N_TRANSACTIONS)   # fresh env, same seed
            m = metrics.setdefault(r.label, {k: [] for k in
                                   ("success", "outage_success", "regret",
                                    "failed_share", "recovery_a")})
            m["success"].append(r.success_rate)
            m["outage_success"].append(r.outage_success_rate)
            m["regret"].append(r.total_regret)
            m["failed_share"].append(traffic_to_failed_gateway(r))
            m["recovery_a"].append(recovery_time(r, outage_a.start, outage_a.gateway))
            if seed == 0:
                example_runs[r.label] = r
        print(f"  seed {seed + 1}/{n_seeds} done", flush=True)
    return metrics, example_runs


def summarise(metrics: dict) -> list:
    """Mean ± std per policy, sorted by overall success rate."""
    rows = []
    static_success = np.mean(metrics["Static priority"]["success"])
    for label, m in metrics.items():
        rows.append({
            "policy": label,
            "success_rate": round(float(np.mean(m["success"])), 4),
            "success_std": round(float(np.std(m["success"])), 4),
            "outage_success_rate": round(float(np.mean(m["outage_success"])), 4),
            "cumulative_regret": round(float(np.mean(m["regret"])), 1),
            "traffic_to_failed_gateway": round(float(np.mean(m["failed_share"])), 4),
            "recovery_txns_outage1": int(np.median(m["recovery_a"])),
            # business framing: extra successful payments per 1M transactions vs static rules
            "extra_success_per_million_vs_static":
                int(round((np.mean(m["success"]) - static_success) * 1_000_000)),
        })
    return sorted(rows, key=lambda r: r["success_rate"], reverse=True)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def shade_outages(ax):
    for o in default_scenario().outages:
        ax.axvspan(o.start, o.start + o.length, color="red", alpha=0.08)


def plot_rolling_success(runs: dict, labels: list):
    fig, ax = plt.subplots(figsize=(11, 4.5))
    kernel = np.ones(SMOOTH) / SMOOTH
    for label in labels:
        r = runs[label]
        rolling = np.convolve(r.rewards, kernel, mode="valid")
        # x = last transaction of each window, so dips line up with the outage bands
        ax.plot(np.arange(len(rolling)) + SMOOTH, rolling, label=label, lw=1.4)
    shade_outages(ax)
    ax.set(title=f"Rolling success rate ({SMOOTH}-txn window) — red bands = gateway outages",
           xlabel="Transaction #", ylabel="Success rate", ylim=(0.55, 1.0))
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGS / "rolling_success_rate.png", dpi=150)
    plt.close(fig)


def plot_regret(runs: dict):
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for label, r in runs.items():
        ax.plot(np.cumsum(r.regret), label=label, lw=1.3)
    shade_outages(ax)
    ax.set(title="Cumulative regret (lower is better)", xlabel="Transaction #",
           ylabel="Lost success probability vs. perfect oracle")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGS / "cumulative_regret.png", dpi=150)
    plt.close(fig)


def plot_traffic_split(run_result, names: list):
    """Which gateway the policy chose over time (stacked share of traffic)."""
    fig, ax = plt.subplots(figsize=(11, 4.5))
    kernel = np.ones(SMOOTH) / SMOOTH
    shares = [np.convolve((run_result.arms == g).astype(float), kernel, mode="valid")
              for g in range(len(names))]
    ax.stackplot(np.arange(len(shares[0])) + SMOOTH, shares, labels=names, alpha=0.85)
    for o in default_scenario().outages:
        ax.axvline(o.start, color="k", ls="--", lw=1)
        ax.text(o.start + 200, 1.02, f"{names[o.gateway]} down", fontsize=9)
    ax.set(title=f"Traffic split chosen by {run_result.label}", xlabel="Transaction #",
           ylabel="Share of traffic", ylim=(0, 1.08))
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.tight_layout()
    fig.savefig(FIGS / "traffic_split.png", dpi=150)
    plt.close(fig)


def plot_bars(rows: list):
    fig, ax = plt.subplots(figsize=(9, 4.8))
    labels = [r["policy"] for r in rows][::-1]
    overall = [r["success_rate"] for r in rows][::-1]
    outage = [r["outage_success_rate"] for r in rows][::-1]
    y = np.arange(len(labels))
    ax.barh(y + 0.2, overall, height=0.4, label="Overall", color="#4C72B0")
    ax.barh(y - 0.2, outage, height=0.4, label="During outages", color="#DD8452")
    ax.set_yticks(y, labels)
    ax.set(xlim=(0.6, 1.0), xlabel="Transaction success rate",
           title="Success rate by routing policy (mean over seeds)")
    ax.legend()
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGS / "policy_comparison.png", dpi=150)
    plt.close(fig)


def write_markdown(rows: list, n_seeds: int):
    lines = [
        f"# Benchmark results ({n_seeds} seeds × {N_TRANSACTIONS:,} transactions, window = {WINDOW})\n",
        "| Policy | Success rate | During outages | Traffic to failed gateway | "
        "Recovery (txns, median) | Cum. regret | Extra successes / 1M vs static |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        rec = "never" if r["recovery_txns_outage1"] >= N_TRANSACTIONS - 15_000 else r["recovery_txns_outage1"]
        lines.append(
            f"| {r['policy']} | {r['success_rate']:.2%} ± {r['success_std']:.2%} | "
            f"{r['outage_success_rate']:.2%} | {r['traffic_to_failed_gateway']:.1%} | {rec} | "
            f"{r['cumulative_regret']:,.0f} | {r['extra_success_per_million_vs_static']:+,} |")
    (REPORTS / "results.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="3 seeds instead of 10")
    args = parser.parse_args()
    n_seeds = 3 if args.quick else 10
    FIGS.mkdir(parents=True, exist_ok=True)

    print(f"Running {n_seeds} seeds × {N_TRANSACTIONS:,} transactions ...")
    metrics, runs = evaluate(n_seeds)
    rows = summarise(metrics)

    (REPORTS / "results.json").write_text(json.dumps(rows, indent=2))
    write_markdown(rows, n_seeds)
    names = default_scenario().names
    plot_rolling_success(runs, ["Static priority", "Thompson Sampling",
                                f"Thompson Sampling (W={WINDOW})"])
    plot_regret(runs)
    plot_traffic_split(runs[f"Thompson Sampling (W={WINDOW})"], names)
    plot_bars(rows)

    print("\n" + (REPORTS / "results.md").read_text())
    print(f"Saved: {REPORTS / 'results.json'}, {REPORTS / 'results.md'}, figures in {FIGS}")


if __name__ == "__main__":
    main()

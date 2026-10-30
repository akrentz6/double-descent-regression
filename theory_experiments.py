import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from double_descent.experiments import DEFAULT_RATIOS, add_common_arguments, run_sweep
from double_descent.theory import asymptotic_risk, finite_sample_risk


def summarise_results(rows):
    """
    Arithmetic trial means and theory evaluated at the actual integer p.
    """
    summary = []
    for n, p in sorted({(row["n"], row["p"]) for row in rows}):
        selected = [row for row in rows if row["n"] == n and row["p"] == p]
        first = selected[0]
        excess = np.array([row["excess_mse"] for row in selected])
        trials = len(selected)
        result = {
            key: first[key] for key in
            ("n", "p", "ratio", "n_test", "seed", "noise_std", "signal_norm")
        }
        result.update({
            "trials": trials,
            "empirical_excess_mean": float(np.mean(excess)),
            "empirical_excess_se": float(np.std(excess, ddof=1) / np.sqrt(trials))
                if trials > 1 else float("nan"),
            "population_risk_mean": float(np.mean([r["coefficient_error"] for r in selected])),
        })
        for name, prediction in (
            ("finite", finite_sample_risk(n, p, first["noise_std"], first["signal_norm"])),
            ("asymptotic", asymptotic_risk(p / n, first["noise_std"], first["signal_norm"])),
        ):
            result.update({f"{name}_{key}": float(value) for key, value in prediction.items()})
        summary.append(result)
    return summary


def run_comparison(sample_sizes=(10, 30, 100), n_test=2000, ratios=DEFAULT_RATIOS,
                   trials=200, seed=0, noise_std=0.5, signal_norm=1.0):
    sample_sizes, ratios = list(sample_sizes), list(ratios)
    rows = []
    for n in sorted(set(sample_sizes)):
        rows.extend(run_sweep(
            n=n, n_test=n_test, ratios=ratios, trials=trials, seed=seed,
            noise_std=noise_std, signal_norm=signal_norm,
        ))
        print(f"n={n}: {len({r['p'] for r in rows if r['n'] == n})} feature counts, "
              f"{trials} trials each", flush=True)
    return rows, summarise_results(rows)


def save_comparison(rows, summary, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for filename, results in (("theory_trials.csv", rows), ("theory_summary.csv", summary)):
        path = output_dir / filename
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(results[0]))
            writer.writeheader()
            writer.writerows(results)
        paths.append(path)

    sizes = sorted({row["n"] for row in summary})
    fig, axes = plt.subplots(1, len(sizes), figsize=(5 * len(sizes), 4.8),
                             sharey=True, squeeze=False)
    for ax, n in zip(axes[0], sizes):
        selected = sorted([row for row in summary if row["n"] == n], key=lambda r: r["p"])
        ratios = np.array([row["ratio"] for row in selected])
        first = selected[0]
        ax.set_yscale("symlog", linthresh=0.01)
        # Include a NaN at the singularity to avoid joining the two branches.
        grid = np.unique(np.append(np.linspace(ratios.min(), ratios.max(), 600), 1.0))
        grid = grid[(grid >= ratios.min()) & (grid <= ratios.max())]
        prediction = asymptotic_risk(grid, first["noise_std"], first["signal_norm"])
        for key, label, color, style in (
            ("total", "Asymptotic total", "black", "-"),
            ("bias", "Theoretical bias", "tab:green", ":"),
            ("variance", "Asymptotic variance", "tab:blue", "--"),
        ):
            values = np.where(np.isfinite(prediction[key]), prediction[key], np.nan)
            ax.plot(grid, values, color=color, linestyle=style, label=label, linewidth=1.4)
        # Evaluate every integer p so even sparse sweeps show the finite-n
        # curve accurately, with a gap where its expectation is infinite.
        feature_grid = np.arange(selected[0]["p"], selected[-1]["p"] + 1)
        finite = finite_sample_risk(n, feature_grid, first["noise_std"], first["signal_norm"])["total"]
        ax.plot(feature_grid / n, np.where(np.isfinite(finite), finite, np.nan),
                color="tab:orange", linestyle="--", label="Finite-n expected total")
        empirical = [row["empirical_excess_mean"] for row in selected]
        ax.plot(ratios, empirical,
                color="tab:red", marker="o", markersize=3, linestyle="none",
                label="Empirical excess risk", zorder=3)
        ax.axvline(1, color="gray", linestyle=":", linewidth=1)
        ax.set(xlabel="p / n", title=f"n = {n}", xlim=(ratios.min(), ratios.max()))
        ax.grid(axis="y", alpha=0.2)
    axes[0, 0].set_ylabel("Mean excess prediction risk (symlog)")
    axes[0, 0].set_ylim(0, 100)
    first = summary[0]
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3)
    fig.tight_layout(rect=(0, 0.13, 1, 0.87))
    fig.suptitle(f"Ridgeless regression\n"
                 f"noise std = {first['noise_std']:g}, "
                 f"signal norm = {first['signal_norm']:g}", y=0.98)
    plot_path = output_dir / "theory_comparison.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    return [*paths, plot_path]


def main():
    parser = argparse.ArgumentParser()
    add_common_arguments(parser, multiple_sizes=True)
    parser.add_argument("--noise-std", type=float, default=0.5)
    args = parser.parse_args()
    rows, summary = run_comparison(
        sample_sizes=args.sample_sizes, n_test=args.n_test, ratios=args.ratios,
        trials=args.trials, seed=args.seed, noise_std=args.noise_std,
        signal_norm=args.signal_norm,
    )
    for path in save_comparison(rows, summary, args.output_dir):
        print(f"Saved {path}")


if __name__ == "__main__":
    main()

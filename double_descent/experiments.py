import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from .data import make_coefficients, sample_linear_data
from .estimators import fit_linear
from .metrics import design_diagnostics, mse

DEFAULT_RATIOS = list(np.linspace(0.1, 3.0, 59)) # 0.05 intervals from 0.1 to 3.0

def run_sweep(n=100, n_test=2000, ratios=DEFAULT_RATIOS, trials=5, seed=0,
              noise_std=0.5, signal_norm=1.0, alpha=0.0):
    """
    Returns one result dict per feature count and trial.
    """
    if n < 1 or n_test < 1 or trials < 1 or seed < 0:
        raise ValueError("sample sizes and trials must be positive; seed must be nonnegative")
    for name, value in (("noise_std", noise_std), ("signal_norm", signal_norm), ("alpha", alpha)):
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    ratios = list(ratios)
    if not ratios or any(not np.isfinite(r) or r <= 0 for r in ratios):
        raise ValueError("ratios must be nonempty, finite, and positive")
    feature_counts = sorted({int(round(n * r)) for r in ratios})
    if feature_counts[0] < 1:
        raise ValueError("ratios must round to at least one feature")

    rows = []
    for p in feature_counts:
        for trial in range(trials):
            rng = np.random.default_rng(np.random.SeedSequence([seed, trial, p]))
            beta = make_coefficients(p, rng, signal_norm)
            X, y = sample_linear_data(n, beta, rng, noise_std)
            X_test, y_test = sample_linear_data(n_test, beta, rng, noise_std)
            beta_hat = fit_linear(X, y, alpha)
            rows.append({
                "n": n, "n_test": n_test, "p": p, "ratio": p / n,
                "trial": trial, "seed": seed, "noise_std": noise_std,
                "signal_norm": signal_norm, "alpha": alpha,
                "train_mse": mse(y, X @ beta_hat),
                "test_mse": mse(y_test, X_test @ beta_hat),
                "excess_mse": mse(X_test @ beta, X_test @ beta_hat),
                "coefficient_error": float(np.linalg.norm(beta_hat - beta) ** 2),
                "coefficient_norm": float(np.linalg.norm(beta_hat)),
                **design_diagnostics(X),
            })
    return rows


def save_results(rows, output_dir, name, group_by=None):
    """
    Saves raw trials to CSV and plot mean test MSE against actual p/n.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{name}.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.set_yscale("symlog", linthresh=0.01)
    groups = sorted({row[group_by] for row in rows}) if group_by else [None]
    for group in groups:
        selected = [r for r in rows if group_by is None or r[group_by] == group]
        ratios = sorted({r["ratio"] for r in selected})
        means = [np.mean([r["test_mse"] for r in selected if r["ratio"] == ratio])
                 for ratio in ratios]
        label = f"{group_by}={group:g}" if group_by else "OLS / minimum norm"
        ax.plot(ratios, means, label=label)
    ax.axvline(1, color="gray", linestyle="--", linewidth=1)
    ax.set_ylim(bottom=max(0, ax.get_ylim()[0]))
    ax.set(xlabel="p / n", ylabel="Mean test MSE",
           title="Double descent: linear regression")
    ax.legend()
    fig.tight_layout()
    plot_path = output_dir / f"{name}.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    return csv_path, plot_path

def add_common_arguments(parser):
    parser.add_argument("--n", type=int, default=100, help="training sample count")
    parser.add_argument("--n-test", type=int, default=2000)
    parser.add_argument("--ratios", type=float, nargs="+", default=DEFAULT_RATIOS,
                        help="p/n values, rounded to integer feature counts")
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--signal-norm", type=float, default=1.0)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))


def run_and_save(parser, args, settings, name, group_by=None):
    """
    Run a few (noise_std, alpha) settings with shared simulation options.
    """
    rows = []
    for noise_std, alpha in settings:
        try:
            results = run_sweep(
                n=args.n, n_test=args.n_test, ratios=args.ratios, trials=args.trials,
                seed=args.seed, signal_norm=args.signal_norm, noise_std=noise_std, alpha=alpha,
            )
        except ValueError as error:
            parser.error(str(error))
        rows.extend(results)
        print(f"noise_std={noise_std:g}, alpha={alpha:g}: {len(results)} fits")
    for path in save_results(rows, args.output_dir, name, group_by):
        print(f"Saved {path}")

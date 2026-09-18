"""Convergence and repeatability analysis for Dynamic Robust ACO.

Produces:
1. convergence_single_run.{png,pdf}
2. convergence_dynamic_epochs.{png,pdf}
3. convergence_memory_comparison.{png,pdf}
4. convergence_30_runs_ci95.{png,pdf}
5. convergence_anytime.{png,pdf}
6. convergence_components.{png,pdf}
7. convergence_summary.csv

Use --runs 30 for article experiments. A smaller value is useful for a quick test.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple
import numpy as np
import matplotlib.pyplot as plt

from dynamic_robust_aco import ChargingAssignmentProblem, DynamicRobustACO
from example_charging_swarm import build_cache, make_cost_matrix
from scalable_scenario import make_scalable_scenario


@dataclass
class EpochData:
    problem0: ChargingAssignmentProblem
    problem1_factory: object


def make_optimizer(max_time=None, forgetting=0.15):
    return DynamicRobustACO(
        epoch=50,
        pop_size=35,
        alpha=1.0,
        heuristic_power=2.5,
        evaporation=0.20,
        forgetting=forgetting,
        q_pheromone=8.0,
        max_time=max_time,
    )


def build_epoch_data(number_of_robots=50, number_of_stations=6, rows=10, cols=10, seed=42):
    scenario = make_scalable_scenario(
        number_of_robots=number_of_robots,
        number_of_stations=number_of_stations,
        rows=rows,
        cols=cols,
        seed=seed,
    )
    return scenario.problem_0, scenario.problem_1_factory


def history_snapshot(model):
    return {
        "fitness": np.asarray(model.history.list_global_best_fit, dtype=float),
        "time": np.asarray(model.history.list_epoch_time, dtype=float),
        "fe": np.asarray(model.history.list_function_evaluations, dtype=int),
        "components": {
            k: np.asarray(v, dtype=float)
            for k, v in model.history.objective_components.items()
        },
    }


def pad_histories(histories):
    """Right-pad best-so-far curves with their final value."""
    n = max(len(h) for h in histories)
    return np.vstack([np.pad(h, (0, n-len(h)), mode="edge") for h in histories])


def normalize(history, eps=1e-12):
    h = np.asarray(history, dtype=float)
    delta = h[0] - h[-1]
    return np.zeros_like(h) if abs(delta) <= eps else (h - h[-1]) / delta


def save(fig, output, stem):
    fig.savefig(output / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(output / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def summary(values):
    a = np.asarray(values, dtype=float)
    return {
        "mean": float(a.mean()), "std": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
        "median": float(np.median(a)), "best": float(a.min()), "worst": float(a.max()),
    }


def main(runs=30, output_dir="convergence_results", robots=50, stations=6, rows=10, cols=10, seed=42):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    p0, p1_factory = build_epoch_data(robots, stations, rows, cols, seed)

    # Single dynamic pair. Preserve t0 history before t1 overwrites it.
    dynamic = make_optimizer(max_time=None)
    r0 = dynamic.solve(p0, seed=42)
    h0 = history_snapshot(dynamic)
    p1 = p1_factory(r0.assignment)
    r1_memory = dynamic.solve(p1, seed=43)
    h1_memory = history_snapshot(dynamic)

    # Same t1 instance without transferred pheromone.
    cold = make_optimizer(max_time=None)
    r1_cold = cold.solve(p1, seed=43)
    h1_cold = history_snapshot(cold)

    # 1: single run.
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(h0["fitness"], linewidth=2)
    ax.set(xlabel="ACO iteration", ylabel="Best objective value",
           title="Dynamic Robust ACO convergence at decision epoch t0")
    ax.grid(True, alpha=0.3)
    fig.tight_layout(); save(fig, output, "convergence_single_run")

    # 2: raw and normalized dynamic epochs.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(h0["fitness"], label=r"Epoch $t_0$", linewidth=2)
    axes[0].plot(h1_memory["fitness"], label=r"Epoch $t_1$", linewidth=2)
    axes[0].set(xlabel="ACO iteration", ylabel="Best objective value", title="Raw convergence")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[1].plot(normalize(h0["fitness"]), label=r"Epoch $t_0$", linewidth=2)
    axes[1].plot(normalize(h1_memory["fitness"]), label=r"Epoch $t_1$", linewidth=2)
    axes[1].set(xlabel="ACO iteration", ylabel="Normalized remaining improvement", title="Normalized convergence")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)
    fig.tight_layout(); save(fig, output, "convergence_dynamic_epochs")

    # 3: memory comparison at t1.
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(h1_memory["fitness"], label="Dynamic ACO with pheromone transfer", linewidth=2)
    ax.plot(h1_cold["fitness"], label="ACO initialized from scratch", linewidth=2, linestyle="--")
    ax.set(xlabel="ACO iteration", ylabel="Best objective value",
           title=r"Effect of pheromone transfer at epoch $t_1$")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout(); save(fig, output, "convergence_memory_comparison")

    # 4: repeated runs for the same static epoch, mean and approximate 95% CI.
    repeated, final_fit, runtime, switches = [], [], [], []
    repeated_time, repeated_fe = [], []
    for seed in range(1, runs + 1):
        model = make_optimizer(max_time=None)
        result = model.solve(p0, seed=seed)
        hist = history_snapshot(model)
        repeated.append(hist["fitness"])
        repeated_time.append(hist["time"])
        repeated_fe.append(hist["fe"])
        final_fit.append(result.fitness)
        runtime.append(result.runtime_seconds)
        switches.append(result.components.get("switch_count", 0.0))
    H = pad_histories(repeated)
    mean, std = H.mean(axis=0), H.std(axis=0, ddof=1) if runs > 1 else np.zeros(H.shape[1])
    ci = 1.96 * std / np.sqrt(runs)
    x = np.arange(H.shape[1])
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x, mean, linewidth=2, label="Mean best objective")
    ax.fill_between(x, mean-ci, mean+ci, alpha=0.22, label="Approximate 95% CI")
    ax.set(xlabel="ACO iteration", ylabel="Best objective value",
           title=f"Convergence over {runs} independent runs")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout(); save(fig, output, "convergence_30_runs_ci95")

    # 5: anytime and function-evaluation views for dynamic memory vs cold start.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(1000*h1_memory["time"], h1_memory["fitness"], label="With memory", linewidth=2)
    axes[0].plot(1000*h1_cold["time"], h1_cold["fitness"], label="Cold start", linewidth=2, linestyle="--")
    axes[0].set(xlabel="Elapsed optimization time (ms)", ylabel="Best objective value", title="Anytime convergence")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[1].plot(h1_memory["fe"], h1_memory["fitness"], label="With memory", linewidth=2)
    axes[1].plot(h1_cold["fe"], h1_cold["fitness"], label="Cold start", linewidth=2, linestyle="--")
    axes[1].set(xlabel="Objective-function evaluations", ylabel="Best objective value", title="Evaluation-budget convergence")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)
    fig.tight_layout(); save(fig, output, "convergence_anytime")

    # 6: objective components at t1 with memory.
    fig, ax = plt.subplots(figsize=(7, 4))
    for name in ("nominal", "robust", "congestion", "switching"):
        ax.plot(h1_memory["components"][name], label=name.capitalize(), linewidth=2)
    ax.set(xlabel="ACO iteration", ylabel="Objective component", title=r"Best-solution components at epoch $t_1$")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout(); save(fig, output, "convergence_components")

    stats_fit = summary(final_fit)
    stats_time = summary(np.asarray(runtime) * 1000.0)
    with (output / "convergence_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "mean", "std", "median", "best", "worst"])
        writer.writerow(["final_fitness", *[stats_fit[k] for k in ("mean", "std", "median", "best", "worst")]])
        writer.writerow(["runtime_ms", *[stats_time[k] for k in ("mean", "std", "median", "best", "worst")]])
    print(f"Generated convergence figures and CSV in: {output.resolve()}")
    print(f"Runs: {runs}; mean final fitness={stats_fit['mean']:.4f}; mean runtime={stats_time['mean']:.2f} ms")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--output", default="convergence_results")
    parser.add_argument("--robots", type=int, default=50)
    parser.add_argument("--stations", type=int, default=6)
    parser.add_argument("--rows", type=int, default=10)
    parser.add_argument("--cols", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(runs=args.runs, output_dir=args.output, robots=args.robots, stations=args.stations, rows=args.rows, cols=args.cols, seed=args.seed)

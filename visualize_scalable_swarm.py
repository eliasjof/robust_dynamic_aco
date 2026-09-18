"""Visualize a scalable swarm without overcrowding labels."""
from __future__ import annotations
import argparse
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
from dynamic_robust_aco import DynamicRobustACO
from scalable_scenario import make_scalable_scenario
from example_scalable_swarm import make_optimizer


def draw(ax, scenario, robot_vertices, soc, problem, result, title, previous=None):
    cmap = plt.get_cmap("tab10")
    colors = {sid: cmap(i % 10) for i, sid in enumerate(problem.station_ids)}
    for u, v, _ in scenario.edges:
        ax.plot([scenario.positions[u][0], scenario.positions[v][0]],
                [scenario.positions[u][1], scenario.positions[v][1]],
                color="#DDDDDD", linewidth=0.45, zorder=1)
    changed = set()
    if previous:
        changed = {rid for rid, sid in result.assignment.items() if previous.get(rid) != sid}
    show_labels = len(problem.robot_ids) <= 30
    rng = np.random.default_rng(123)
    for i, rid in enumerate(problem.robot_ids):
        v = robot_vertices[rid]
        x, y = scenario.positions[v]
        x += rng.uniform(-0.13, 0.13); y += rng.uniform(-0.13, 0.13)
        sid = result.assignment[rid]
        edge = "red" if rid in changed else "black"
        size = 18 + 70 * soc[i]
        ax.scatter(x, y, s=size, color=colors[sid], alpha=0.78,
                   edgecolor=edge, linewidth=1.2 if rid in changed else 0.25, zorder=4)
        if show_labels:
            ax.text(x, y, rid[1:], fontsize=5, ha="center", va="center", color="white")
    loads = Counter(result.assignment.values())
    for j, (sid, vertex) in enumerate(scenario.stations.items()):
        x, y = scenario.positions[vertex]
        total = int(problem.current_load[j] + loads[sid])
        cap = int(problem.total_capacity[j])
        ax.scatter(x, y, s=170, marker="s", color=colors[sid], edgecolor="black", zorder=6)
        ax.text(x, y + 0.30, f"{sid}\n{total}/{cap}", fontsize=7,
                ha="center", fontweight="bold", color=colors[sid])
    ax.set_title(title)
    ax.set_aspect("equal"); ax.axis("off")
    ax.text(0.01, 0.01, f"R={len(problem.robot_ids)}, S={len(problem.station_ids)}, "
            f"fitness={result.fitness:.2f}, switches={int(result.components.get('switch_count', 0))}",
            transform=ax.transAxes, fontsize=8,
            bbox=dict(facecolor="white", edgecolor="#AAAAAA", boxstyle="round,pad=0.2"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--robots", type=int, default=50)
    parser.add_argument("--stations", type=int, default=6)
    parser.add_argument("--rows", type=int, default=10)
    parser.add_argument("--cols", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    sc = make_scalable_scenario(args.robots, args.stations, args.rows, args.cols, seed=args.seed)
    model = make_optimizer(args.robots)
    r0 = model.solve(sc.problem_0, seed=args.seed)
    p1 = sc.problem_1_factory(r0.assignment)
    r1 = model.solve(p1, seed=args.seed + 1)
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), constrained_layout=True)
    draw(axes[0], sc, sc.robot_vertices_0, sc.soc_0, sc.problem_0, r0, "Decision epoch t0")
    draw(axes[1], sc, sc.robot_vertices_1, sc.soc_1, p1, r1, "Decision epoch t1", r0.assignment)
    fig.suptitle("Scalable dynamic robust charging assignment")
    fig.savefig("scalable_swarm_visualization.png", dpi=300, bbox_inches="tight")
    fig.savefig("scalable_swarm_visualization.pdf", bbox_inches="tight")
    print("Saved scalable_swarm_visualization.png and .pdf")


if __name__ == "__main__":
    main()

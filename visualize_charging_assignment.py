"""Visual demonstration of the robust dynamic charging-assignment problem."""
from __future__ import annotations

from collections import Counter
import numpy as np
import matplotlib.pyplot as plt

from dynamic_robust_aco import ChargingAssignmentProblem, DynamicRobustACO
from example_charging_swarm import build_cache, make_cost_matrix, recover_route


def build_scenario():
    edges = [
        (0, 1, 2.0), (1, 2, 2.0), (2, 3, 3.0), (3, 4, 2.0),
        (4, 5, 2.0), (5, 6, 3.0), (6, 7, 2.0), (7, 8, 2.0),
        (1, 4, 4.0), (2, 5, 4.0), (3, 6, 4.0), (4, 7, 4.0),
        (0, 3, 6.0), (5, 8, 5.0),
    ]
    graph = {v: [] for v in range(9)}
    for u, v, w in edges:
        graph[u].append((v, w))
        graph[v].append((u, w))
    positions = {
        0: (0.0, 0.0), 1: (1.0, 0.8), 2: (2.0, 0.0),
        3: (3.0, 0.9), 4: (4.0, 0.0), 5: (5.0, 0.9),
        6: (6.0, 0.0), 7: (7.0, 0.8), 8: (8.0, 0.0),
    }
    stations = {"CS-A": 0, "CS-B": 4, "CS-C": 8}
    return graph, edges, positions, stations


def solve_epochs():
    graph, edges, positions, stations = build_scenario()
    station_ids = tuple(stations)
    distance_cache, next_hops = build_cache(graph, stations)
    robot_vertices_0 = {
        "R1": 1, "R2": 2, "R3": 3, "R4": 5,
        "R5": 6, "R6": 7, "R7": 2, "R8": 6,
    }
    robot_ids = tuple(robot_vertices_0)
    costs_0 = make_cost_matrix(robot_vertices_0, station_ids, distance_cache)
    dev_0 = 0.15 * costs_0 + 0.25
    soc_0 = np.array([0.34, 0.39, 0.42, 0.32, 0.36, 0.30, 0.45, 0.38])
    budgets_0 = 26.0 * soc_0 - 1.0
    total_capacity = np.array([4, 4, 4])
    load_0 = np.array([1, 0, 1])
    p0 = ChargingAssignmentProblem(
        robot_ids, station_ids, costs_0, dev_0, budgets_0,
        total_capacity - load_0, load_0, total_capacity,
        gamma=2.5, congestion_weight=0.20, switching_weight=1.50,
        epsilon=0.05,
    )
    model = DynamicRobustACO(
        epoch=50, pop_size=35, alpha=1.0, heuristic_power=2.5,
        evaporation=0.20, forgetting=0.15, q_pheromone=8.0,
        max_time=0.25,
    )
    r0 = model.solve(p0, seed=42)

    robot_vertices_1 = dict(robot_vertices_0)
    robot_vertices_1.update({"R2": 3, "R4": 6, "R6": 8})
    costs_1 = make_cost_matrix(robot_vertices_1, station_ids, distance_cache)
    dev_1 = 0.18 * costs_1 + 0.25
    soc_1 = np.maximum(soc_0 - 0.035, 0.10)
    budgets_1 = 26.0 * soc_1 - 1.0
    load_1 = np.array([2, 1, 0])
    p1 = ChargingAssignmentProblem(
        robot_ids, station_ids, costs_1, dev_1, budgets_1,
        total_capacity - load_1, load_1, total_capacity,
        previous_assignment=r0.assignment, gamma=2.5,
        congestion_weight=0.20, switching_weight=1.50, epsilon=0.05,
    )
    r1 = model.solve(p1, seed=43)
    return (graph, edges, positions, stations, next_hops,
            robot_vertices_0, soc_0, p0, r0,
            robot_vertices_1, soc_1, p1, r1)


def draw_epoch(ax, title, edges, positions, stations, next_hops,
               robot_vertices, soc, problem, result, previous=None):
    colors = {"CS-A": "#0072B2", "CS-B": "#009E73", "CS-C": "#D55E00"}
    for u, v, _ in edges:
        x = [positions[u][0], positions[v][0]]
        y = [positions[u][1], positions[v][1]]
        ax.plot(x, y, color="#D0D0D0", linewidth=1.2, zorder=1)
    for vertex, (x, y) in positions.items():
        ax.scatter(x, y, s=24, color="#777777", zorder=2)
        ax.text(x, y - 0.18, f"v{vertex}", fontsize=7, ha="center", color="#555555")

    # Draw the selected nominal paths first.
    for i, rid in enumerate(problem.robot_ids):
        sid = result.assignment[rid]
        route = recover_route(robot_vertices[rid], sid, stations, next_hops)
        for a, b in zip(route[:-1], route[1:]):
            ax.plot([positions[a][0], positions[b][0]],
                    [positions[a][1], positions[b][1]],
                    color=colors[sid], linewidth=2.6, alpha=0.48, zorder=3)

    # Stations.
    assigned = Counter(result.assignment.values())
    for j, (sid, vertex) in enumerate(stations.items()):
        x, y = positions[vertex]
        total = int(problem.current_load[j] + assigned[sid])
        cap = int(problem.total_capacity[j])
        ax.scatter(x, y, s=250, marker="s", color=colors[sid], edgecolor="black", zorder=6)
        ax.text(x, y + 0.29, f"{sid}\nload {total}/{cap}", ha="center", va="bottom",
                fontsize=8, fontweight="bold", color=colors[sid])

    # Robots, offset when sharing a vertex.
    occupancy = Counter()
    changed = set()
    if previous is not None:
        changed = {rid for rid, sid in result.assignment.items() if previous.get(rid) != sid}
    for i, rid in enumerate(problem.robot_ids):
        v = robot_vertices[rid]
        idx = occupancy[v]
        occupancy[v] += 1
        dx = (-0.20 if idx % 2 == 0 else 0.20) * (1 + idx // 2)
        x, y = positions[v]
        sid = result.assignment[rid]
        edge = "#CC0000" if rid in changed else "black"
        width = 2.2 if rid in changed else 0.8
        ax.scatter(x + dx, y + 0.16, s=105, marker="o", color=colors[sid],
                   edgecolor=edge, linewidth=width, zorder=7)
        ax.text(x + dx, y + 0.16, rid[1:], color="white", fontsize=7,
                ha="center", va="center", fontweight="bold", zorder=8)
        ax.text(x + dx, y + 0.36, f"{100*soc[i]:.0f}%", fontsize=6.5,
                ha="center", color="#222222", zorder=8)

    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.text(0.01, 0.02,
            f"fitness={result.fitness:.2f} | nominal={result.components['nominal']:.2f} | "
            f"robust={result.components['robust']:.2f} | congestion={result.components['congestion']:.2f}",
            transform=ax.transAxes, fontsize=7.5,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#AAAAAA"))
    ax.set_xlim(-0.65, 8.65)
    ax.set_ylim(-0.55, 1.75)
    ax.set_aspect("equal")
    ax.axis("off")


def main():
    (graph, edges, positions, stations, next_hops,
     rv0, soc0, p0, r0, rv1, soc1, p1, r1) = solve_epochs()

    fig, axes = plt.subplots(2, 1, figsize=(12, 7.6), constrained_layout=True)
    draw_epoch(axes[0], "Decision epoch t0: initial robust assignment",
               edges, positions, stations, next_hops, rv0, soc0, p0, r0)
    draw_epoch(axes[1], "Decision epoch t1: dynamic reassignment after state changes",
               edges, positions, stations, next_hops, rv1, soc1, p1, r1,
               previous=r0.assignment)
    fig.suptitle("Dynamic robust charging-station assignment for a robot swarm",
                 fontsize=14, fontweight="bold")
    fig.text(0.5, 0.002,
             "Line and robot color indicate the assigned station. Red robot borders identify changed assignments at t1.",
             ha="center", fontsize=9)
    fig.savefig("charging_assignment_visualization.png", dpi=220, bbox_inches="tight")
    fig.savefig("charging_assignment_visualization.pdf", bbox_inches="tight")
    print("Saved charging_assignment_visualization.png and .pdf")


if __name__ == "__main__":
    main()

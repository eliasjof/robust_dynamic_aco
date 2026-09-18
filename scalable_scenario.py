"""Scalable instance generator for dynamic robot charging assignment."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Hashable, Tuple
import numpy as np

from dynamic_robust_aco import ChargingAssignmentProblem
from example_charging_swarm import build_cache, make_cost_matrix


@dataclass
class ScalableScenario:
    graph: dict
    edges: list
    positions: dict
    stations: dict
    distance_cache: dict
    next_hops: dict
    robot_vertices_0: dict
    robot_vertices_1: dict
    soc_0: np.ndarray
    soc_1: np.ndarray
    problem_0: ChargingAssignmentProblem
    problem_1_factory: object


def create_grid_graph(rows: int, cols: int, edge_cost: float = 1.0):
    graph = {}
    positions = {}
    edges = []

    def vid(row, col):
        return row * cols + col

    for row in range(rows):
        for col in range(cols):
            vertex = vid(row, col)
            graph[vertex] = []
            positions[vertex] = (float(col), float(rows - 1 - row))

    for row in range(rows):
        for col in range(cols):
            vertex = vid(row, col)
            if col + 1 < cols:
                neighbor = vid(row, col + 1)
                graph[vertex].append((neighbor, edge_cost))
                graph[neighbor].append((vertex, edge_cost))
                edges.append((vertex, neighbor, edge_cost))
            if row + 1 < rows:
                neighbor = vid(row + 1, col)
                graph[vertex].append((neighbor, edge_cost))
                graph[neighbor].append((vertex, edge_cost))
                edges.append((vertex, neighbor, edge_cost))
    return graph, edges, positions


def choose_station_vertices(rows: int, cols: int, number_of_stations: int):
    candidates = []
    for row in np.linspace(0, rows - 1, num=max(2, int(np.ceil(np.sqrt(number_of_stations)))), dtype=int):
        for col in np.linspace(0, cols - 1, num=max(2, int(np.ceil(np.sqrt(number_of_stations)))), dtype=int):
            candidates.append(int(row * cols + col))
    # Add central and edge locations before removing duplicates.
    candidates.extend([
        (rows // 2) * cols + cols // 2,
        (rows // 2) * cols,
        (rows // 2) * cols + cols - 1,
    ])
    unique = list(dict.fromkeys(candidates))
    if len(unique) < number_of_stations:
        unique.extend(v for v in range(rows * cols) if v not in unique)
    selected = unique[:number_of_stations]
    return {f"CS-{j + 1}": vertex for j, vertex in enumerate(selected)}


def generate_station_capacities(number_of_robots, number_of_stations, capacity_margin=1.30):
    required = int(np.ceil(number_of_robots * capacity_margin))
    base, remainder = divmod(required, number_of_stations)
    capacities = np.full(number_of_stations, base, dtype=int)
    capacities[:remainder] += 1
    return capacities


def generate_current_load(total_capacity, max_occupancy=0.10, seed=44):
    rng = np.random.default_rng(seed)
    upper = np.floor(max_occupancy * total_capacity).astype(int)
    return np.array([rng.integers(0, value + 1) for value in upper], dtype=int)


def generate_robots(number_of_robots, vertices, seed=42):
    rng = np.random.default_rng(seed)
    robot_ids = tuple(f"R{i + 1}" for i in range(number_of_robots))
    selected = rng.choice(tuple(vertices), size=number_of_robots, replace=True)
    return robot_ids, {rid: int(vertex) for rid, vertex in zip(robot_ids, selected)}


def move_robots(robot_vertices, graph, move_probability=0.35, seed=45):
    rng = np.random.default_rng(seed)
    moved = {}
    for rid, vertex in robot_vertices.items():
        if rng.random() < move_probability and graph[vertex]:
            options = [neighbor for neighbor, _ in graph[vertex]]
            moved[rid] = int(rng.choice(options))
        else:
            moved[rid] = vertex
    return moved


def ensure_individual_reachability(costs, deviations, proposed_budgets, slack=0.50):
    minimum_required = np.min(costs + deviations, axis=1)
    return np.maximum(proposed_budgets, minimum_required + slack)




def ensure_k_station_reachability(
    costs,
    deviations,
    proposed_budgets,
    minimum_reachable_stations=2,
    slack=0.50,
):
    """Adjust synthetic budgets so each robot can reach at least k stations.

    This helper is only for generating feasible benchmark instances. Physical
    experiments must use measured energy budgets and an emergency policy.
    """
    robust_costs = np.asarray(costs, dtype=float) + np.asarray(deviations, dtype=float)
    k = min(max(int(minimum_reachable_stations), 1), robust_costs.shape[1])
    kth_required = np.sort(robust_costs, axis=1)[:, k - 1] + float(slack)
    return np.maximum(np.asarray(proposed_budgets, dtype=float), kth_required)


def make_scalable_scenario(
    number_of_robots=50,
    number_of_stations=6,
    rows=10,
    cols=10,
    capacity_margin=1.30,
    gamma_fraction=0.25,
    seed=42,
):
    graph, edges, positions = create_grid_graph(rows, cols)
    stations = choose_station_vertices(rows, cols, number_of_stations)
    station_ids = tuple(stations)
    distance_cache, next_hops = build_cache(graph, stations)

    robot_ids, rv0 = generate_robots(number_of_robots, graph.keys(), seed=seed)
    rng = np.random.default_rng(seed + 1)
    soc0 = rng.uniform(0.25, 0.50, size=number_of_robots)
    c0 = make_cost_matrix(rv0, station_ids, distance_cache)
    d0 = 0.15 * c0 + 0.25
    L0 = ensure_individual_reachability(c0, d0, 40.0 * soc0 - 2.0)

    total_capacity = generate_station_capacities(
        number_of_robots, number_of_stations, capacity_margin
    )
    q0 = generate_current_load(total_capacity, max_occupancy=0.08, seed=seed + 2)
    if int((total_capacity - q0).sum()) < number_of_robots:
        total_capacity += int(np.ceil((number_of_robots - (total_capacity - q0).sum()) / number_of_stations))
    p0 = ChargingAssignmentProblem(
        robot_ids=robot_ids,
        station_ids=station_ids,
        nominal_cost=c0,
        deviation=d0,
        travel_budget=L0,
        residual_capacity=total_capacity - q0,
        current_load=q0,
        total_capacity=total_capacity,
        gamma=gamma_fraction * number_of_robots,
        congestion_weight=0.20,
        switching_weight=1.50,
        epsilon=0.05,
    )

    rv1 = move_robots(rv0, graph, move_probability=0.35, seed=seed + 3)
    soc1 = np.maximum(soc0 - rng.uniform(0.02, 0.05, size=number_of_robots), 0.10)
    c1 = make_cost_matrix(rv1, station_ids, distance_cache)
    d1 = 0.18 * c1 + 0.25
    L1 = ensure_individual_reachability(c1, d1, 40.0 * soc1 - 2.0)
    q1 = generate_current_load(total_capacity, max_occupancy=0.10, seed=seed + 4)
    if int((total_capacity - q1).sum()) < number_of_robots:
        q1[:] = 0

    def problem_1_factory(previous_assignment):
        return ChargingAssignmentProblem(
            robot_ids=robot_ids,
            station_ids=station_ids,
            nominal_cost=c1,
            deviation=d1,
            travel_budget=L1,
            residual_capacity=total_capacity - q1,
            current_load=q1,
            total_capacity=total_capacity,
            previous_assignment=previous_assignment,
            gamma=gamma_fraction * number_of_robots,
            congestion_weight=0.20,
            switching_weight=1.50,
            epsilon=0.05,
        )

    return ScalableScenario(
        graph, edges, positions, stations, distance_cache, next_hops,
        rv0, rv1, soc0, soc1, p0, problem_1_factory,
    )

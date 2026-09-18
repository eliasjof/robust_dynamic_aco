"""Two-epoch charging assignment example using DynamicRobustACO."""
from __future__ import annotations

import heapq
from typing import Dict, Hashable, List, Tuple
import numpy as np

from dynamic_robust_aco import ChargingAssignmentProblem, DynamicRobustACO, MEALPY_AVAILABLE


def station_rooted_dijkstra(graph, station_vertex):
    """Return distances and next-hop pointers toward one fixed station."""
    dist = {v: float("inf") for v in graph}
    next_hop = {station_vertex: station_vertex}
    dist[station_vertex] = 0.0
    heap = [(0.0, station_vertex)]
    while heap:
        dv, v = heapq.heappop(heap)
        if dv != dist[v]:
            continue
        for u, weight in graph[v]:
            candidate = dv + weight
            if candidate < dist[u]:
                dist[u] = candidate
                # Search is rooted at station; from u, move first to v.
                next_hop[u] = v
                heapq.heappush(heap, (candidate, u))
    return dist, next_hop


def build_cache(graph, stations):
    distance_cache, next_hop_cache = {}, {}
    for station_id, vertex in stations.items():
        distances, hops = station_rooted_dijkstra(graph, vertex)
        distance_cache[station_id] = distances
        next_hop_cache[station_id] = hops
    return distance_cache, next_hop_cache


def recover_route(start_vertex, station_id, stations, next_hop_cache):
    goal = stations[station_id]
    route = [start_vertex]
    current = start_vertex
    seen = {current}
    while current != goal:
        current = next_hop_cache[station_id][current]
        if current in seen:
            raise RuntimeError("Cycle detected in next-hop cache")
        route.append(current)
        seen.add(current)
    return route


def make_cost_matrix(robot_vertices, station_ids, distance_cache):
    return np.array([
        [distance_cache[sid][robot_vertices[rid]] for sid in station_ids]
        for rid in robot_vertices
    ], dtype=float)


def print_result(title, result, problem, robot_vertices, stations, next_hops):
    print(f"\n{title}")
    print("-" * len(title))
    print(f"MEALPY installed: {MEALPY_AVAILABLE}")
    print(f"Fitness: {result.fitness:.4f}")
    print(f"Components: {result.components}")
    print(f"Runtime: {1000.0 * result.runtime_seconds:.2f} ms")
    for rid, sid in result.assignment.items():
        i = problem.robot_ids.index(rid)
        j = problem.station_ids.index(sid)
        route = recover_route(robot_vertices[rid], sid, stations, next_hops)
        print(
            f"{rid} -> {sid}: nominal={problem.nominal_cost[i,j]:.1f}, "
            f"deviation={problem.deviation[i,j]:.1f}, "
            f"budget={problem.travel_budget[i]:.1f}, route={route}"
        )


def main():
    # Undirected navigation graph; weights may represent distance or energy.
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

    stations = {"CS-A": 0, "CS-B": 4, "CS-C": 8}
    station_ids = tuple(stations)
    distance_cache, next_hops = build_cache(graph, stations)

    robot_vertices_1 = {
        "R1": 1, "R2": 2, "R3": 3, "R4": 5,
        "R5": 6, "R6": 7, "R7": 2, "R8": 6,
    }
    robot_ids = tuple(robot_vertices_1)
    costs_1 = make_cost_matrix(robot_vertices_1, station_ids, distance_cache)
    deviations_1 = 0.15 * costs_1 + 0.25  # Test bound; calibrate from data in a real system.
    # Travel budget derived from SoC after an operational reserve.
    soc_1 = np.array([0.34, 0.39, 0.42, 0.32, 0.36, 0.30, 0.45, 0.38])
    budgets_1 = 26.0 * soc_1 - 1.0
    current_load_1 = np.array([1, 0, 1])
    total_capacity = np.array([4, 4, 4])
    residual_1 = total_capacity - current_load_1

    problem_1 = ChargingAssignmentProblem(
        robot_ids=robot_ids,
        station_ids=station_ids,
        nominal_cost=costs_1,
        deviation=deviations_1,
        travel_budget=budgets_1,
        residual_capacity=residual_1,
        current_load=current_load_1,
        total_capacity=total_capacity,
        previous_assignment={},
        gamma=2.5,
        congestion_weight=0.20,
        switching_weight=1.50,
        epsilon=0.05,
    )

    optimizer = DynamicRobustACO(
        epoch=50,
        pop_size=35,
        alpha=1.0,
        heuristic_power=2.5,
        evaporation=0.20,
        forgetting=0.15,
        q_pheromone=8.0,
        max_time=0.25,
    )
    result_1 = optimizer.solve(problem_1, seed=42)
    print_result("Decision epoch t0", result_1, problem_1, robot_vertices_1, stations, next_hops)

    # Epoch t1: robots moved, workloads changed, and previous assignment is retained.
    robot_vertices_2 = dict(robot_vertices_1)
    robot_vertices_2.update({"R2": 3, "R4": 6, "R6": 8})
    costs_2 = make_cost_matrix(robot_vertices_2, station_ids, distance_cache)
    deviations_2 = 0.18 * costs_2 + 0.25
    soc_2 = np.maximum(soc_1 - 0.035, 0.10)
    budgets_2 = 26.0 * soc_2 - 1.0
    current_load_2 = np.array([2, 1, 0])
    residual_2 = total_capacity - current_load_2

    problem_2 = ChargingAssignmentProblem(
        robot_ids=robot_ids,
        station_ids=station_ids,
        nominal_cost=costs_2,
        deviation=deviations_2,
        travel_budget=budgets_2,
        residual_capacity=residual_2,
        current_load=current_load_2,
        total_capacity=total_capacity,
        previous_assignment=result_1.assignment,
        gamma=2.5,
        congestion_weight=0.20,
        switching_weight=1.50,
        epsilon=0.05,
    )
    # Same optimizer instance: valid pheromone is transferred to t1.
    result_2 = optimizer.solve(problem_2, seed=43)
    print_result("Decision epoch t1", result_2, problem_2, robot_vertices_2, stations, next_hops)
    print(f"\nAssignment changes at t1: {int(result_2.components['switch_count'])}")


if __name__ == "__main__":
    main()

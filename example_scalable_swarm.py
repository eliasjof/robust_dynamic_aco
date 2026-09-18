"""Run a scalable dynamic charging-assignment instance."""
from __future__ import annotations
import argparse
from collections import Counter
from dynamic_robust_aco import DynamicRobustACO
from scalable_scenario import make_scalable_scenario


def make_optimizer(robots, max_time=None):
    if robots <= 20:
        epoch, ants = 75, 40
    elif robots <= 50:
        epoch, ants = 100, 60
    elif robots <= 100:
        epoch, ants = 150, 100
    else:
        epoch, ants = 180, 120
    return DynamicRobustACO(
        epoch=epoch, pop_size=ants, alpha=1.0, heuristic_power=2.5,
        evaporation=0.20, forgetting=0.15, q_pheromone=8.0,
        max_time=max_time,
    )


def report(epoch_name, result, problem):
    counts = Counter(result.assignment.values())
    print(f"\n{epoch_name}: fitness={result.fitness:.3f}; runtime={1000*result.runtime_seconds:.1f} ms")
    print(f"components={result.components}")
    print("new assignments per station:", dict(counts))
    print("residual capacity:", dict(zip(problem.station_ids, problem.residual_capacity.tolist())))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--robots", type=int, default=50)
    parser.add_argument("--stations", type=int, default=6)
    parser.add_argument("--rows", type=int, default=10)
    parser.add_argument("--cols", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-time", type=float, default=None)
    args = parser.parse_args()

    scenario = make_scalable_scenario(
        number_of_robots=args.robots, number_of_stations=args.stations,
        rows=args.rows, cols=args.cols, seed=args.seed,
    )
    model = make_optimizer(args.robots, max_time=args.max_time)
    result0 = model.solve(scenario.problem_0, seed=args.seed)
    report("t0", result0, scenario.problem_0)

    problem1 = scenario.problem_1_factory(result0.assignment)
    result1 = model.solve(problem1, seed=args.seed + 1)
    report("t1", result1, problem1)
    print(f"assignment changes at t1: {int(result1.components['switch_count'])}")


if __name__ == "__main__":
    main()

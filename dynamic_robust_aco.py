"""Dynamic Robust Ant Colony Optimizer for charging-station assignment.

The class exposes a MEALPY-style ``solve(problem, seed=None)`` API and, when
MEALPY is installed, inherits from ``mealpy.optimizer.Optimizer``.  The search
itself is combinatorial: pheromones are indexed by persistent
(robot_id, station_id) pairs and are transferred between decision epochs.

This is research code. Validate uncertainty bounds, energy conversion, and
real-time deadlines on the target robotic platform before deployment.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Dict, Hashable, Iterable, List, Mapping, Optional, Sequence, Tuple
import math
import time
import numpy as np

try:
    from mealpy.optimizer import Optimizer as _MealpyOptimizer
    MEALPY_AVAILABLE = True
except ImportError:  # Makes the example runnable without the optional package.
    MEALPY_AVAILABLE = False

    class _MealpyOptimizer:
        """Small fallback exposing the attributes used by this implementation."""
        def __init__(self, **kwargs):
            self.name = kwargs.get("name", self.__class__.__name__)
            self.g_best = None
            self.history = None


@dataclass
class ChargingAssignmentProblem:
    """State of one supervisory decision epoch.

    All values in ``nominal_cost``, ``deviation`` and ``travel_budget`` must
    have consistent units, e.g. energy, time, or equivalent distance.
    """

    robot_ids: Sequence[Hashable]
    station_ids: Sequence[Hashable]
    nominal_cost: np.ndarray             # shape (R, S), c_ij
    deviation: np.ndarray                # shape (R, S), d_ij
    travel_budget: np.ndarray            # shape (R,), L_i
    residual_capacity: np.ndarray        # shape (S,), A_j
    current_load: Optional[np.ndarray] = None  # shape (S,), q_j
    total_capacity: Optional[np.ndarray] = None # shape (S,), C_j
    previous_assignment: Mapping[Hashable, Hashable] = field(default_factory=dict)
    gamma: float = 0.0
    congestion_weight: float = 1.0
    switching_weight: float = 0.0
    epsilon: float = 1.0e-6

    def __post_init__(self) -> None:
        self.robot_ids = tuple(self.robot_ids)
        self.station_ids = tuple(self.station_ids)
        self.nominal_cost = np.asarray(self.nominal_cost, dtype=float)
        self.deviation = np.asarray(self.deviation, dtype=float)
        self.travel_budget = np.asarray(self.travel_budget, dtype=float)
        self.residual_capacity = np.asarray(self.residual_capacity, dtype=int)
        r, s = len(self.robot_ids), len(self.station_ids)
        if self.nominal_cost.shape != (r, s):
            raise ValueError(f"nominal_cost must have shape {(r, s)}")
        if self.deviation.shape != (r, s):
            raise ValueError(f"deviation must have shape {(r, s)}")
        if self.travel_budget.shape != (r,):
            raise ValueError(f"travel_budget must have shape {(r,)}")
        if self.residual_capacity.shape != (s,):
            raise ValueError(f"residual_capacity must have shape {(s,)}")
        if np.any(self.nominal_cost < 0) or np.any(self.deviation < 0):
            raise ValueError("Costs and deviations must be nonnegative")
        if np.any(self.travel_budget < 0) or np.any(self.residual_capacity < 0):
            raise ValueError("Budgets and capacities must be nonnegative")
        if not 0.0 <= self.gamma <= float(r):
            raise ValueError("gamma must belong to [0, number of robots]")
        if self.current_load is None:
            self.current_load = np.zeros(s, dtype=int)
        else:
            self.current_load = np.asarray(self.current_load, dtype=int)
        if self.total_capacity is None:
            self.total_capacity = self.current_load + self.residual_capacity
        else:
            self.total_capacity = np.asarray(self.total_capacity, dtype=int)
        if self.current_load.shape != (s,) or self.total_capacity.shape != (s,):
            raise ValueError("current_load and total_capacity must have shape (S,)")
        if np.any(self.total_capacity <= 0):
            raise ValueError("Each total station capacity must be positive")
        if np.any(self.current_load + self.residual_capacity > self.total_capacity):
            raise ValueError("current_load + residual_capacity exceeds total_capacity")

    @property
    def feasible_mask(self) -> np.ndarray:
        """Arc (i,j) is safe under the local worst-case deviation."""
        return (self.nominal_cost + self.deviation <= self.travel_budget[:, None]) & \
               (self.residual_capacity[None, :] > 0)


@dataclass
class ACOResult:
    """Result object with MEALPY-like ``solution`` and ``target.fitness``."""

    solution: np.ndarray
    fitness: float
    assignment: Dict[Hashable, Hashable]
    components: Dict[str, float]
    runtime_seconds: float
    epochs_completed: int
    feasible: bool = True

    def __post_init__(self) -> None:
        self.target = SimpleNamespace(fitness=float(self.fitness), objectives=[float(self.fitness)])


class DynamicRobustACO(_MealpyOptimizer):
    """Dynamic ACO for robust, capacitated robot-to-charger assignment.

    Parameters follow common ACO notation. ``pop_size`` is the number of ants
    and ``epoch`` is the number of colony iterations per decision epoch.
    """

    def __init__(
        self,
        epoch: int = 60,
        pop_size: int = 40,
        alpha: float = 1.0,
        heuristic_power: float = 2.0,
        evaporation: float = 0.20,
        forgetting: float = 0.15,
        q_pheromone: float = 1.0,
        tau0: float = 1.0,
        tau_min: float = 1.0e-4,
        tau_max: float = 20.0,
        robust_heuristic_fraction: bool = True,
        max_construction_restarts: int = 20,
        max_time: Optional[float] = None,
        name: str = "DynamicRobustACO",
        **kwargs,
    ) -> None:
        super().__init__(name=name, **kwargs)
        if epoch < 1 or pop_size < 2:
            raise ValueError("epoch >= 1 and pop_size >= 2 are required")
        if alpha < 0 or heuristic_power < 0:
            raise ValueError("alpha and heuristic_power must be nonnegative")
        if not 0 < evaporation <= 1 or not 0 <= forgetting <= 1:
            raise ValueError("evaporation in (0,1], forgetting in [0,1]")
        if not 0 < tau_min <= tau0 <= tau_max:
            raise ValueError("Require 0 < tau_min <= tau0 <= tau_max")
        self.epoch = int(epoch)
        self.pop_size = int(pop_size)
        self.alpha = float(alpha)
        self.heuristic_power = float(heuristic_power)
        self.evaporation = float(evaporation)
        self.forgetting = float(forgetting)
        self.q_pheromone = float(q_pheromone)
        self.tau0 = float(tau0)
        self.tau_min = float(tau_min)
        self.tau_max = float(tau_max)
        self.robust_heuristic_fraction = bool(robust_heuristic_fraction)
        self.max_construction_restarts = int(max_construction_restarts)
        self.max_time = max_time
        self.pheromone: Dict[Tuple[Hashable, Hashable], float] = {}
        self.history_best: List[float] = []
        self.history_time: List[float] = []
        self.history_fe: List[int] = []
        self.history_components: Dict[str, List[float]] = {}
        self.history_epoch_best: List[float] = []
        self.history_epoch_mean: List[float] = []
        self.history_epoch_std: List[float] = []
        self.history_diversity: List[float] = []
        self.history_hamming: List[float] = []
        self.history_pheromone_entropy: List[float] = []
        self.history_tau_ratio: List[float] = []
        self.history_iter: Dict[str, List[float]] = {}
        self.g_best: Optional[ACOResult] = None

    def reset_memory(self) -> None:
        """Forget all pheromone transferred from previous decision epochs."""
        self.pheromone.clear()

    def _transfer_pheromone(self, problem: ChargingAssignmentProblem) -> None:
        """Keep valid robot-station memories and softly forget old evidence."""
        valid = set()
        mask = problem.feasible_mask
        for i, rid in enumerate(problem.robot_ids):
            for j, sid in enumerate(problem.station_ids):
                if mask[i, j]:
                    key = (rid, sid)
                    old = self.pheromone.get(key, self.tau0)
                    self.pheromone[key] = (1.0 - self.forgetting) * old + self.forgetting * self.tau0
                    valid.add(key)
        self.pheromone = {key: value for key, value in self.pheromone.items() if key in valid}

    @staticmethod
    def _maximum_b_matching(problem: ChargingAssignmentProblem) -> Optional[np.ndarray]:
        """Find one feasible unit-demand assignment by DFS over station slots."""
        mask = problem.feasible_mask
        slots: List[int] = []
        for j, cap in enumerate(problem.residual_capacity):
            slots.extend([j] * int(cap))
        if len(slots) < len(problem.robot_ids):
            return None
        matched_robot = [-1] * len(slots)
        assignment = np.full(len(problem.robot_ids), -1, dtype=int)
        order = sorted(range(len(problem.robot_ids)), key=lambda i: int(mask[i].sum()))

        def augment(i: int, seen: set) -> bool:
            possible_slots = [k for k, j in enumerate(slots) if mask[i, j]]
            possible_slots.sort(key=lambda k: problem.nominal_cost[i, slots[k]])
            for slot_idx in possible_slots:
                if slot_idx in seen:
                    continue
                seen.add(slot_idx)
                other = matched_robot[slot_idx]
                if other == -1 or augment(other, seen):
                    matched_robot[slot_idx] = i
                    assignment[i] = slots[slot_idx]
                    return True
            return False

        for i in order:
            if not augment(i, set()):
                return None
        return assignment

    @staticmethod
    def _robust_term(active_deviations: np.ndarray, gamma: float) -> float:
        """Bertsimas-Sim budget term using descending order statistics."""
        if gamma <= 0 or active_deviations.size == 0:
            return 0.0
        values = np.sort(np.asarray(active_deviations, dtype=float))[::-1]
        whole = min(int(math.floor(gamma)), len(values))
        fraction = gamma - math.floor(gamma)
        value = float(values[:whole].sum())
        if fraction > 0 and whole < len(values):
            value += float(fraction * values[whole])
        return value

    @staticmethod
    def evaluate(problem: ChargingAssignmentProblem, assignment: np.ndarray) -> Tuple[float, Dict[str, float]]:
        r = len(problem.robot_ids)
        if assignment.shape != (r,):
            raise ValueError("assignment has wrong shape")
        nominal = float(problem.nominal_cost[np.arange(r), assignment].sum())
        active_dev = problem.deviation[np.arange(r), assignment]
        robust = DynamicRobustACO._robust_term(active_dev, problem.gamma)
        new_load = np.bincount(assignment, minlength=len(problem.station_ids))
        total_load = problem.current_load + new_load
        
        if np.any(total_load > problem.total_capacity):
            return math.inf, {"nominal": nominal, "robust": robust, "congestion": math.inf, "queue": math.inf, "waiting": math.inf, "switching": math.inf}
        chargers = np.asarray(getattr(problem, "charger_capacity", problem.total_capacity), dtype=int)
        charge_time = np.asarray(getattr(problem, "charging_time", np.ones(len(problem.station_ids))), dtype=float)
        queue_weight = float(getattr(problem, "queue_weight", 1.0))
        wait_weight = float(getattr(problem, "waiting_weight", 1.0))
        soc = np.asarray(getattr(problem, "soc", np.ones(r)), dtype=float)
        queue_cost = 0.0
        waiting_cost = 0.0
        for j in range(len(problem.station_ids)):
            members = [i for i in range(r) if int(assignment[i]) == j]
            members.sort(key=lambda i: (float(soc[i]), i))
            occupied = int(problem.current_load[j])
            for rank, i in enumerate(members, start=1):
                absolute_position = occupied + rank
                wait_cycles = max(0, (absolute_position - 1) // max(1, int(chargers[j])))
                wait = wait_cycles * float(charge_time[j])
                queue_cost += queue_weight * max(0, absolute_position - int(chargers[j]))
                waiting_cost += wait_weight * wait
        utilization = total_load / problem.total_capacity
        barrier = utilization / (1.0 - utilization + problem.epsilon)
        congestion = float(problem.congestion_weight * barrier.sum())
        switching_count = 0
        for i, rid in enumerate(problem.robot_ids):
            previous = problem.previous_assignment.get(rid)
            if previous is not None and previous != problem.station_ids[int(assignment[i])]:
                switching_count += 1
        switching = float(problem.switching_weight * switching_count)
        components = {
            "nominal": nominal,
            "robust": robust,
            "congestion": congestion,
            "queue": float(queue_cost),
            "waiting": float(waiting_cost),
            "switching": switching,
            "switch_count": float(switching_count),
        }
        return nominal + robust + congestion + queue_cost + waiting_cost + switching, components

    def _construct_ant(self, problem: ChargingAssignmentProblem, rng: np.random.Generator) -> Optional[np.ndarray]:
        mask = problem.feasible_mask
        r = len(problem.robot_ids)
        min_slack = np.where(mask, problem.travel_budget[:, None] - problem.nominal_cost - problem.deviation, np.inf).min(axis=1)
        order = sorted(range(r), key=lambda i: (int(mask[i].sum()), float(min_slack[i])))
        gamma_fraction = problem.gamma / max(r, 1) if self.robust_heuristic_fraction else 0.0

        for _ in range(self.max_construction_restarts):
            remaining = problem.residual_capacity.astype(int).copy()
            assignment = np.full(r, -1, dtype=int)
            success = True
            for i in order:
                candidates = np.flatnonzero(mask[i] & (remaining > 0))
                if candidates.size == 0:
                    success = False
                    break
                scores = []
                for j in candidates:
                    projected_load = problem.current_load[j] + (problem.residual_capacity[j] - remaining[j]) + 1
                    u = projected_load / problem.total_capacity[j]
                    delta_barrier = u / (1.0 - u + problem.epsilon)
                    switch = float(problem.previous_assignment.get(problem.robot_ids[i]) not in (None, problem.station_ids[j]))
                    guide = problem.nominal_cost[i, j] + gamma_fraction * problem.deviation[i, j]
                    guide += problem.congestion_weight * delta_barrier + problem.switching_weight * switch
                    eta = 1.0 / max(guide, problem.epsilon)
                    tau = self.pheromone[(problem.robot_ids[i], problem.station_ids[j])]
                    scores.append((tau ** self.alpha) * (eta ** self.heuristic_power))
                probs = np.asarray(scores, dtype=float)
                if not np.all(np.isfinite(probs)) or probs.sum() <= 0:
                    probs = np.full(len(candidates), 1.0 / len(candidates))
                else:
                    probs /= probs.sum()
                j = int(rng.choice(candidates, p=probs))
                assignment[i] = j
                remaining[j] -= 1
            if success:
                return assignment
        return None

    def _deposit(self, problem: ChargingAssignmentProblem, assignment: np.ndarray, fitness: float) -> None:
        for key in list(self.pheromone):
            self.pheromone[key] = max(self.tau_min, (1.0 - self.evaporation) * self.pheromone[key])
        amount = self.q_pheromone / max(float(fitness), 1.0e-12)
        for i, j in enumerate(assignment):
            key = (problem.robot_ids[i], problem.station_ids[int(j)])
            self.pheromone[key] = min(self.tau_max, self.pheromone[key] + amount)

    def _pheromone_diagnostics(self, problem):
        """Return mean normalized entropy and global max/min pheromone ratio."""
        entropies, all_values = [], []
        mask = problem.feasible_mask
        for i, robot_id in enumerate(problem.robot_ids):
            values = np.asarray([
                self.pheromone[(robot_id, station_id)]
                for j, station_id in enumerate(problem.station_ids) if mask[i, j]
            ], dtype=float)
            if values.size == 0:
                continue
            all_values.extend(values.tolist())
            if values.size == 1:
                entropies.append(0.0)
            else:
                probabilities = values / values.sum()
                entropy = -np.sum(probabilities * np.log(probabilities + 1.0e-12))
                entropies.append(float(entropy / np.log(values.size)))
        if not all_values:
            return 0.0, 1.0
        values = np.asarray(all_values, dtype=float)
        return float(np.mean(entropies)), float(values.max() / max(values.min(), 1.0e-12))

    def solve(
        self,
        problem: ChargingAssignmentProblem,
        seed: Optional[int] = None,
        **kwargs,
    ) -> ACOResult:
        if not isinstance(problem, ChargingAssignmentProblem):
            raise TypeError("problem must be a ChargingAssignmentProblem")
        start = time.perf_counter()
        rng = np.random.default_rng(seed)
        baseline = self._maximum_b_matching(problem)
        if baseline is None:
            raise ValueError("No complete robust capacitated assignment exists for this decision epoch")
        self._transfer_pheromone(problem)
        best_assignment = baseline.copy()
        best_fitness, best_components = self.evaluate(problem, best_assignment)
        
        self.history_iter = {'fitness_iter': [], 'pheromone_mean': []}
        self.history_best = [best_fitness]
        self.history_time = [0.0]
        self.history_fe = [1]
        self.history_epoch_best = []
        self.history_epoch_mean = []
        self.history_epoch_std = []
        self.history_diversity = []
        self.history_hamming = []
        self.history_pheromone_entropy = []
        self.history_tau_ratio = []
        self.history_components = {
            name: [float(best_components.get(name, 0.0))]
            for name in ("nominal", "robust", "congestion", "switching", "switch_count")
        }
        function_evaluations = 1
        epochs_completed = 0

        for ep in range(self.epoch):
            colony = []
            for _ in range(self.pop_size):
                candidate = self._construct_ant(problem, rng)
                if candidate is None:
                    continue
                fitness, components = self.evaluate(problem, candidate)
                function_evaluations += 1
                colony.append((fitness, candidate, components))
            if colony:
                colony.sort(key=lambda item: item[0])
                epoch_fitness, epoch_assignment, epoch_components = colony[0]
                values = np.asarray([item[0] for item in colony], dtype=float)
                self.history_epoch_best.append(float(values.min()))
                self.history_epoch_mean.append(float(values.mean()))
                self.history_epoch_std.append(float(values.std()))
                unique = {tuple(item[1].tolist()) for item in colony}
                self.history_diversity.append(len(unique) / len(colony))
                self.history_hamming.append(float(np.mean([
                    np.mean(item[1] != epoch_assignment) for item in colony
                ])))
                if epoch_fitness < best_fitness:
                    best_fitness = epoch_fitness
                    best_assignment = epoch_assignment.copy()
                    best_components = dict(epoch_components)
                self._deposit(problem, epoch_assignment, epoch_fitness)
            else:
                self.history_epoch_best.append(float(best_fitness))
                self.history_epoch_mean.append(float(best_fitness))
                self.history_epoch_std.append(0.0)
                self.history_diversity.append(0.0)
                self.history_hamming.append(0.0)
            
            # Adicionado para suportar os gráficos de convergência por iteração
            self.history_iter['fitness_iter'].append(float(best_fitness))
            mean_pher = float(np.mean(list(self.pheromone.values()))) if self.pheromone else 0.0
            self.history_iter['pheromone_mean'].append(mean_pher)
            
            entropy, tau_ratio = self._pheromone_diagnostics(problem)
            self.history_pheromone_entropy.append(entropy)
            self.history_tau_ratio.append(tau_ratio)
            self.history_best.append(best_fitness)
            self.history_time.append(time.perf_counter() - start)
            self.history_fe.append(function_evaluations)
            for name in self.history_components:
                self.history_components[name].append(float(best_components.get(name, 0.0)))
            epochs_completed = ep + 1
            if self.max_time is not None and time.perf_counter() - start >= self.max_time:
                break

        assignment_dict = {
            rid: problem.station_ids[int(best_assignment[i])]
            for i, rid in enumerate(problem.robot_ids)
        }
        result = ACOResult(
            solution=best_assignment.astype(float),
            fitness=best_fitness,
            assignment=assignment_dict,
            components=best_components,
            runtime_seconds=time.perf_counter() - start,
            epochs_completed=epochs_completed,
        )
        self.g_best = result
        self.history = SimpleNamespace(
            list_global_best_fit=list(self.history_best),
            list_epoch_time=list(self.history_time),
            list_function_evaluations=list(self.history_fe),
            objective_components={key: list(values) for key, values in self.history_components.items()},
            list_epoch_best_fit=list(self.history_epoch_best),
            list_epoch_mean_fit=list(self.history_epoch_mean),
            list_epoch_std_fit=list(self.history_epoch_std),
            list_population_diversity=list(self.history_diversity),
            list_mean_hamming_distance=list(self.history_hamming),
            list_pheromone_entropy=list(self.history_pheromone_entropy),
            list_pheromone_tau_ratio=list(self.history_tau_ratio),
        )
        return result
# Dynamic Robust ACO compatible with MEALPY

This example implements a discrete, dynamic Ant Colony Optimization method for
robust capacitated robot-to-charging-station assignment.

## Files

- `dynamic_robust_aco.py`: optimizer, problem state, robust objective, capacity
  handling, pheromone transfer, and feasibility test.
- `example_charging_swarm.py`: navigation graph, station-rooted Dijkstra cache,
  eight robots, three charging stations, and two dynamic decision epochs.

## Installation

```bash
python -m pip install "mealpy>=3.1,<3.2" numpy
```

MEALPY is optional for running this example because the module includes a small
fallback base class. When installed, `DynamicRobustACO` inherits from
`mealpy.optimizer.Optimizer` and keeps the familiar `solve(problem, seed=...)`,
`g_best`, `solution`, and `target.fitness` conventions.

## Run

```bash
python example_charging_swarm.py
```

## Important modeling notes

1. `nominal_cost`, `deviation`, and `travel_budget` must use the same unit.
2. The example converts SoC to a test travel budget using a simple linear rule.
   Replace it with a calibrated energy model in the article experiments.
3. `c_ij + d_ij <= L_i` is a hard local survivability filter.
4. The Bertsimas-Sim `gamma` term protects fleet-level aggregate cost.
5. The same optimizer object is reused across epochs so valid pheromone is
   transferred with controlled forgetting.
6. The built-in bipartite b-matching test rejects epochs with no complete
   feasible assignment before ACO is executed.

## Visualize the graph and assignments

```bash
python visualize_charging_assignment.py
```

This creates `charging_assignment_visualization.png` and `.pdf`.

## Generate convergence figures

Quick validation:

```bash
python convergence_analysis.py --runs 5
```

Article experiment with 30 independent runs:

```bash
python convergence_analysis.py --runs 30
```

Outputs are written to `convergence_results/`:

- single-run convergence;
- raw and normalized convergence for two dynamic epochs;
- ACO with pheromone memory versus cold-start ACO;
- mean convergence over repeated runs with approximate 95% confidence interval;
- best objective versus elapsed time and function evaluations;
- objective-component convergence;
- CSV summary containing mean, standard deviation, median, best, and worst values.

The optimizer now records:

```python
optimizer.history.list_global_best_fit
optimizer.history.list_epoch_time
optimizer.history.list_function_evaluations
optimizer.history.objective_components
```

## Scalable experiments

The scalable generator creates a grid graph, distributes fixed charging stations,
generates robot positions and SoC values, scales station capacities, and guarantees
at least one individually reachable station per synthetic robot. The last rule is
only for generating feasible benchmark instances; a real robot without a reachable
station must activate the emergency policy.

Run 50 robots, 6 stations, and a 10x10 graph:

```bash
python example_scalable_swarm.py --robots 50 --stations 6 --rows 10 --cols 10
python visualize_scalable_swarm.py --robots 50 --stations 6 --rows 10 --cols 10
python convergence_analysis.py --runs 30 --robots 50 --stations 6 --rows 10 --cols 10
```

Suggested scalability cases:

```bash
python example_scalable_swarm.py --robots 20  --stations 4  --rows 8  --cols 8
python example_scalable_swarm.py --robots 50  --stations 6  --rows 10 --cols 10
python example_scalable_swarm.py --robots 100 --stations 10 --rows 14 --cols 14
python example_scalable_swarm.py --robots 200 --stations 15 --rows 20 --cols 20
```

For swarms larger than 30 robots, the visualizer hides individual robot labels,
uses point size to represent SoC, colors to represent station assignment, and red
borders to identify reassigned robots.

## Animation t0,...,tN

```bash
python animate_dynamic_swarm.py --robots 30 --stations 5 --rows 10 --cols 10 --epochs 15
```

The script saves numbered PNG frames, an animated GIF, and a CSV time series in
`dynamic_animation/`. Robot motion is a random walk on the graph for demonstration;
replace it with measured robot states for physical experiments.

### Save MP4

MP4 export requires FFmpeg with the `libx264` encoder. Generate GIF, PNG frames,
CSV history, and MP4 together with:

```bash
python animate_dynamic_swarm.py \
  --robots 30 --stations 5 --rows 10 --cols 10 --epochs 15 \
  --save-mp4 --fps 2
```

If FFmpeg is not on the system PATH:

```bash
python animate_dynamic_swarm.py --save-mp4 \
  --ffmpeg /full/path/to/ffmpeg
```

The MP4 is saved as `dynamic_animation/dynamic_swarm_animation.mp4`. The video
uses H.264 and `yuv420p` for broad compatibility. The padding filter guarantees
even frame dimensions, which H.264 commonly requires.

## Feasibility-safe animation generation

The animation now verifies a capacitated bipartite matching before every ACO run.
Synthetic budgets initially guarantee `--minimum-reachable 3` stations per robot.
If joint matching still fails, this number is increased up to all stations. The
default capacity margin is `1.60`, SoC consumption is reduced for long animations,
and robots located at their previously assigned station receive a synthetic charge
increment. These adjustments are benchmark-generation mechanisms, not replacements
for the emergency policy required by a physical system.

Recommended Windows command:

```powershell
python animate_dynamic_swarm.py --robots 30 --stations 5 --epochs 30 `
  --capacity-margin 1.60 --minimum-reachable 3 --save-mp4 --fps 2
```

## Detailed convergence diagnostics

```powershell
python convergence_diagnostics.py --robots 30 --stations 5 --epochs 80 --ants 50
```

Generates fitness dispersion, assignment diversity, Hamming distance, pheromone
entropy, pheromone max/min ratio, an `alpha=1` versus `alpha=0` ablation, and a CSV
with stabilization and exact-gap metrics when enumeration is computationally safe.

## Smooth constant-speed animation

```powershell
python animate_smooth_swarm.py --robots 20 --stations 4 --epochs 12 `
  --frames-per-epoch 6 --fps 6 --save-mp4
```

Each moving robot traverses at most one adjacent graph edge per decision interval.
Intermediate frames use linear arc-length interpolation, so speed is constant and
there are no visual jumps. ACO allocation is recomputed only at integer decision
epochs `t0,...,tN`.

## Label visibility controls

Labels are no longer hidden automatically for large swarms. Use `--label-mode all`
for all labels, `--label-mode critical` for low-SoC/reassigned/robust-budget robots,
or `--label-mode none`. Font size is controlled by `--label-fontsize`.


## Modelo de fila limitada
- `total_capacity`: capacidade de admissao, igual a carregadores + vagas de fila.
- `charger_capacity`: recargas simultaneas.
- `queue_capacity`: limite fisico da fila.
- Prioridade estrita por menor SoC, com indice do robo como desempate deterministico.
- A funcao objetivo inclui `queue` (penalizacao por posicao alem dos carregadores) e `waiting` (ciclos de espera vezes o tempo de recarga).
- Os labels da animacao suave mostram ordem/tamanho da fila e espera; os labels das estacoes mostram ocupacao e fila.

## Running

```
animate_fsm_swarm.py --robots 30 --stations 6 --rows 20 --cols 20 --epochs 35 --aco-epochs 120 --ants 50 --max-time -1.0 --output animation_labels --save-mp4 --fps 2 --frames-per-epoch 10 --capacity-margin 1.0 --chargers-per-station 1 --obstacle-prob 0.2 --battery-threshold 0.35 --seed 430 --internal-heatmap-epoch 30 --dev-base 25.0 --dev-mult 1.5 --q-pheromone 200.0
```

"""
Animação de Enxame Robótico com Máquina de Estados Finita (FSM).
Inclui bloqueio físico de tomada: um robô a carregar (PLUGGED_IN)
só liberta a vaga ao atingir 100%, forçando os restantes a aguardar (WAITING).
"""

from __future__ import annotations
import argparse
import csv
import shutil
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Importação dos módulos do projeto
from dynamic_robust_aco import DynamicRobustACO, ChargingAssignmentProblem
from example_charging_swarm import build_cache, make_cost_matrix
from scalable_scenario import (
    create_grid_graph,
    choose_station_vertices,
    generate_robots,
    generate_station_capacities,
    ensure_k_station_reachability
)
from animate_dynamic_swarm import robust_contributions, find_ffmpeg

# Definição dos Estados Físicos
WORKING = 0
HEADING = 1
WAITING = 2      # Estacionado, à espera da tomada
PLUGGED_IN = 3   # Conectado à recarga


def build_subset_problem(model, active_rids, station_ids, vertices, cache, soc, caps, load, previous, min_reachable, chargers_per_station, gamma_frac=0.25):
    """
    Constrói a instância de otimização apenas para os robôs que precisam de carga.
    """
    if not active_rids:
        return None, 0

    active_soc = np.array([soc[r] for r in active_rids])
    active_vertices = {r: vertices[r] for r in active_rids}
    c_subset = make_cost_matrix(active_vertices, station_ids, cache)

    d_subset = 0.15 * c_subset + 0.25
    proposed_L = 40.0 * active_soc - 2.0
    first_k = min(max(1, min_reachable), len(station_ids))
    
    last_problem = None
    for k in range(first_k, len(station_ids) + 1):
        L = ensure_k_station_reachability(c_subset, d_subset, proposed_L, k, slack=0.50)
        
        problem = ChargingAssignmentProblem(
            robot_ids=active_rids,
            station_ids=station_ids,
            nominal_cost=c_subset,
            deviation=d_subset,
            travel_budget=L,
            residual_capacity=caps - load,
            current_load=load,
            total_capacity=caps,
            previous_assignment=previous,
            gamma=gamma_frac * len(active_rids),
            congestion_weight=25.0,
            switching_weight=10.0,
            epsilon=0.05
        )
        
        # INFORMA O D-ACO SOBRE AS TOMADAS REAIS PARA CÁLCULO PRECISO DAS FILAS
        problem.charger_capacity = np.full(len(station_ids), chargers_per_station)
        
        last_problem = problem
        
        if model._maximum_b_matching(problem) is not None:
            return problem, k
            
    raise RuntimeError("O subconjunto de robôs ativos não possui rotas seguras viáveis.")


def get_next_step(rid, current_v, state, assignment, graph, cache, stations, rng, working_targets, pos):
    """
    Calcula o próximo nó para o qual o robô deve navegar no mapa (grafo).
    """
    if state[rid] in (WAITING, PLUGGED_IN):
        # Imóveis na estação
        return current_v
        
    elif state[rid] == WORKING:
        target_v = working_targets[rid]
        if current_v == target_v:
            return current_v
            
        tx, ty = pos[target_v]
        neighbors = [u for u, _ in graph[current_v]]
        best_neighbor = min(neighbors, key=lambda n: abs(pos[n][0] - tx) + abs(pos[n][1] - ty))
        return best_neighbor
        
    elif state[rid] == HEADING:
        sid = assignment.get(rid)
        if sid is None:
            return current_v
            
        station_v = stations[sid]
        
        def get_dist(n):
            if (n, sid) in cache: return cache[(n, sid)]
            if (sid, n) in cache: return cache[(sid, n)]
            if (n, station_v) in cache: return cache[(n, station_v)]
            if (station_v, n) in cache: return cache[(station_v, n)]
            
            if sid in cache and n in cache[sid]: return cache[sid][n]
            if station_v in cache and n in cache[station_v]: return cache[station_v][n]
            if n in cache and sid in cache[n]: return cache[n][sid]
            if n in cache and station_v in cache[n]: return cache[n][station_v]
            return float('inf')
            
        neighbors = [u for u, _ in graph[current_v]]
        best_neighbor = min(neighbors, key=get_dist)
        return best_neighbor


def interpolate_positions(vertices, target, positions, fraction, jitter):
    out = {}
    for rid, u in vertices.items():
        v = target[rid]
        x0, y0 = positions[u]
        x1, y1 = positions[v]
        dx, dy = jitter[rid]
        out[rid] = (
            (1 - fraction) * x0 + fraction * x1 + dx,
            (1 - fraction) * y0 + fraction * y1 + dy
        )
    return out


def render_fsm(path, epoch, phase, edges, positions, stations, xy, soc, state, problem, result, previous, working_targets, label_fontsize=4.6):
    fig, ax = plt.subplots(figsize=(8, 7))
    cmap = plt.get_cmap('tab10')
    
    station_ids = list(stations.keys())
    colors = {sid: cmap(j % 10) for j, sid in enumerate(station_ids)}
    
    for u, v, _ in edges:
        ax.plot([positions[u][0], positions[v][0]], [positions[u][1], positions[v][1]], color='#eeeeee', lw=0.6, zorder=1)
        
    changed = set()
    if result and problem:
        changed = {r for r, s in result.assignment.items() if previous and previous.get(r) != s}
        
    for rid, coords in xy.items():
        x, y = coords
        robot_soc = soc[rid]
        robot_state = state[rid]
        
        if robot_state == WORKING:
            color = '#999999'
            edgecolor = '#666666'
            linewidth = 0.5
        else:
            sid = result.assignment.get(rid) if result else None
            color = colors[sid] if sid else '#ff0000'
            edgecolor = 'red' if rid in changed else 'black'
            linewidth = 1.8 if rid in changed else 0.5
            
        ax.scatter(x, y, s=20 + 60 * robot_soc, color=color, alpha=0.9, edgecolor=edgecolor, linewidth=linewidth, zorder=4)
        
        # Etiqueta visual indica quem está com a tomada (raio amarelo)
        status_marker = "⚡" if robot_state == PLUGGED_IN else ""
        ax.annotate(
            f'{rid} {status_marker}\n{100*robot_soc:.0f}%',
            (x, y), xytext=(0, 5), textcoords='offset points',
            fontsize=label_fontsize, ha='center', va='bottom', zorder=8, clip_on=False,
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=0.3)
        )
            
    assigned_counts = Counter(result.assignment.values()) if result else Counter()
    for j, (sid, v) in enumerate(stations.items()):
        x, y = positions[v]
        ax.scatter(x, y, s=250, marker='s', color=colors[sid], edgecolor='black', linewidth=1, zorder=7)
        cap = problem.total_capacity[j] if problem else "N/A"
        ax.text(
            x, y + 0.33, f'{sid}\n{assigned_counts[sid]}/{cap}',
            ha='center', fontsize=7, fontweight='bold', color=colors[sid]
        )
        
    ax.set_title(f'FSM Swarm Logistics: $t_{{{epoch}}}$ + {phase:.2f}', fontweight='bold')
    
    if result:
        metrics_text = f"Active D-ACO Agents: {len(problem.robot_ids)} | Switches: {int(result.components['switch_count'])}"
        ax.text(0.01, 0.01, metrics_text, transform=ax.transAxes, fontsize=7.2, bbox=dict(facecolor='white', edgecolor='#aaa', boxstyle='round,pad=0.2'))
        
    ax.set_aspect('equal')
    ax.axis('off')
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches='tight')
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--robots', type=int, default=25)
    p.add_argument('--stations', type=int, default=4)
    p.add_argument('--rows', type=int, default=9)
    p.add_argument('--cols', type=int, default=9)
    p.add_argument('--epochs', type=int, default=60)
    p.add_argument('--frames-per-epoch', type=int, default=5)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--output', default='fsm_animation')
    
    p.add_argument('--aco-epochs', type=int, default=25)
    p.add_argument('--ants', type=int, default=25)
    p.add_argument('--max-time', type=float, default=0.25)
    p.add_argument('--capacity-margin', type=float, default=1.4)
    p.add_argument('--chargers-per-station', type=int, default=1)
    p.add_argument('--minimum-reachable', type=int, default=3)
    p.add_argument('--fps', type=float, default=2.0)
    p.add_argument('--save-mp4', action='store_true')
    p.add_argument('--ffmpeg', default='ffmpeg')

    a = p.parse_args()
    out = Path(a.output)
    frames = out / 'animation_frames'
    frames.mkdir(parents=True, exist_ok=True)
    
    rng = np.random.default_rng(a.seed)
    graph, edges, pos = create_grid_graph(a.rows, a.cols)
    stations = choose_station_vertices(a.rows, a.cols, a.stations)
    station_ids = tuple(stations.keys())
    
    cache, _ = build_cache(graph, stations)
    robot_ids, vertices = generate_robots(a.robots, graph.keys(), a.seed)
    
    soc = {r: rng.uniform(0.35, 1.0) for r in robot_ids}
    state = {r: WORKING for r in robot_ids}
    
    graph_nodes = list(graph.keys())
    working_targets = {r: int(rng.choice(graph_nodes)) for r in robot_ids}
    
    caps = generate_station_capacities(a.robots, a.stations, a.capacity_margin)
    jr = np.random.default_rng(a.seed + 999)
    jitter = {r: (jr.uniform(-0.12, 0.12), jr.uniform(-0.12, 0.12)) for r in robot_ids}
    
    model = DynamicRobustACO(
        epoch=a.aco_epochs, pop_size=a.ants, heuristic_power=2.5,
        evaporation=0.2, forgetting=0.15, q_pheromone=8, max_time=a.max_time
    )
    
    previous_assignment = {}
    paths = []
    frame = 0
    
    for k in range(a.epochs):
        # 1. Atualização Física Base
        for r in robot_ids:
            if state[r] == WORKING:
                soc[r] -= rng.uniform(0.06, 0.12)
                if soc[r] < 0.35:
                    state[r] = HEADING
            elif state[r] == HEADING:
                soc[r] -= rng.uniform(0.010, 0.020)
                sid = previous_assignment.get(r)
                if sid is not None and vertices[r] == stations[sid]:
                    state[r] = WAITING  # Chega e aguarda vaga

        # 2. Lógica Rigorosa de Fila de Espera e Tomadas
        for sid, station_v in stations.items():
            # Filtra quem está nesta estação específica
            waiting = [r for r in robot_ids if state[r] == WAITING and vertices[r] == station_v]
            plugged = [r for r in robot_ids if state[r] == PLUGGED_IN and vertices[r] == station_v]
            
            # Tomadas livres para novos robôs
            free_spots = a.chargers_per_station - len(plugged)
            
            # Admite robôs da fila de espera para as tomadas livres com base no menor SoC
            if free_spots > 0 and waiting:
                waiting.sort(key=lambda r: soc[r])
                for r in waiting[:free_spots]:
                    state[r] = PLUGGED_IN
                    plugged.append(r)
            
            # Carrega apenas quem está ativamente com a tomada
            for r in plugged:
                soc[r] = min(1.0, soc[r] + 0.15)
                # Liberta a tomada e vai trabalhar
                if soc[r] >= 1.0:
                    state[r] = WORKING
                    working_targets[r] = int(rng.choice(graph_nodes))
                    if r in previous_assignment: del previous_assignment[r]

        # 3. Renovação Dinâmica de Waypoints (Garante que ninguém fica parado a trabalhar)
        for r in robot_ids:
            if state[r] == WORKING and vertices[r] == working_targets[r]:
                working_targets[r] = int(rng.choice(graph_nodes))

        # 4. Execução do D-ACO para o subconjunto ativo
        active_rids = [r for r in robot_ids if state[r] in (HEADING, WAITING, PLUGGED_IN)]
        load = np.zeros(a.stations, dtype=int)
        
        problem, kused = build_subset_problem(
            model, active_rids, station_ids, vertices, cache,
            soc, caps, load, previous_assignment, a.minimum_reachable, a.chargers_per_station
        )
        
        result = None
        if problem:
            result = model.solve(problem, seed=a.seed + k)
            previous_assignment = dict(result.assignment)
            
        # 5. Determinação do próximo passo
        target = {}
        for r in robot_ids:
            target[r] = get_next_step(
                r, vertices[r], state, previous_assignment,
                graph, cache, stations, rng, working_targets, pos
            )
            
        # 6. Interpolação e Renderização visual
        for sub in range(a.frames_per_epoch):
            fraction = sub / a.frames_per_epoch
            xy = interpolate_positions(vertices, target, pos, fraction, jitter)
            fp = frames / f'frame_{frame:05d}.png'
            
            render_fsm(
                fp, k, fraction, edges, pos, stations, xy, soc, state,
                problem, result, previous_assignment, working_targets
            )
            paths.append(fp)
            frame += 1
            
        vertices = target
        
    print(f'Geração de frames concluída. Guardados {len(paths)} frames em {out}')

    images = [Image.open(x).convert('RGB') for x in paths]
    duration = int(round(1000 / a.fps))
    images[0].save(out / 'fsm_swarm_animation.gif', save_all=True, append_images=images[1:], duration=duration, loop=0, optimize=False)
    for im in images: im.close()

    if a.save_mp4:
        exe = find_ffmpeg(a.ffmpeg)
        mp4 = out / 'fsm_swarm_animation.mp4'
        subprocess.run([
            exe, '-y', '-framerate', str(a.fps),
            '-i', str(frames / 'frame_%05d.png'),
            '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2',
            '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
            '-movflags', '+faststart', str(mp4)
        ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"Vídeo guardado em: {mp4}")

if __name__ == '__main__':
    main()
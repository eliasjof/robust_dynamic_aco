"""
Animação de Enxame Robótico com Máquina de Estados Finita (FSM) e Extração Académica.
REGRAS FÍSICAS RESTRITAS:
- Task-Aware Routing (Look-ahead): Otimização prevê a distância da recarga até a próxima tarefa.
- Soft Constraint: Prevenção de Crashes. O D-ACO penaliza a inanição mas não aborta a simulação.
- Envelope de Bertsimas-Sim: Cronometragem com Acoplamento de Perímetro (-1 hop).
- Normalização de Probabilidade: Heatmaps exibem a % de Confiança do D-ACO.
- Colisão Zero Absoluta: Prevenção de nós ocupados e cruzamento em X (diagonais).
"""

from __future__ import annotations
import argparse
import csv
import heapq
import shutil
import subprocess
from collections import Counter
from pathlib import Path
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from dynamic_robust_aco import DynamicRobustACO, ChargingAssignmentProblem
from animate_dynamic_swarm import robust_contributions, find_ffmpeg

WORKING = 0
HEADING = 1
WAITING = 2      
PLUGGED_IN = 3   


def create_custom_grid_graph(rows, cols, num_stations, obstacle_prob=0.15, seed=42):
    rng = np.random.default_rng(seed)
    is_obstacle = rng.random((rows, cols)) < obstacle_prob
    
    all_coords = [(r, c) for r in range(rows) for c in range(cols)]
    station_idx = rng.choice(len(all_coords), size=num_stations, replace=False)
    station_coords = [all_coords[i] for i in station_idx]
    
    for sr, sc in station_coords:
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                nr, nc = sr + dr, sc + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    is_obstacle[nr, nc] = False  
    
    pos = {}
    for r in range(rows):
        for c in range(cols):
            if not is_obstacle[r, c]:
                v = r * cols + c
                pos[v] = (c, r)

    raw_graph = {v: [] for v in pos}
    directions = [
        (0, 1, 1.0), (1, 0, 1.0), (0, -1, 1.0), (-1, 0, 1.0),
        (1, 1, 1.414), (1, -1, 1.414), (-1, 1, 1.414), (-1, -1, 1.414)
    ]
    
    for r in range(rows):
        for c in range(cols):
            if is_obstacle[r, c]: 
                continue
            v = r * cols + c
            for dr, dc, w in directions:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and not is_obstacle[nr, nc]:
                    if abs(dr) == 1 and abs(dc) == 1:
                        if is_obstacle[r + dr, c] or is_obstacle[r, c + dc]:
                            continue
                    
                    nv = nr * cols + nc
                    raw_graph[v].append((nv, w))

    visited = set()
    largest_cc = set()
    for node in raw_graph:
        if node not in visited:
            cc = set()
            queue = [node]
            while queue:
                curr = queue.pop(0)
                if curr not in visited:
                    visited.add(curr)
                    cc.add(curr)
                    queue.extend([n for n, weight in raw_graph[curr]])
            if len(cc) > len(largest_cc):
                largest_cc = cc

    graph = {v: [edge for edge in raw_graph[v] if edge[0] in largest_cc] for v in largest_cc}
    final_pos = {v: pos[v] for v in largest_cc}
    
    edges = []
    for v in graph:
        for nv, w in graph[v]:
            if v < nv:
                edges.append((v, nv, w))
                
    obstacle_coords = [(c, r) for r in range(rows) for c in range(cols) if is_obstacle[r, c]]
    
    station_nodes = [r * cols + c for r, c in station_coords]
    station_nodes = [n for n in station_nodes if n in largest_cc]
    
    return graph, edges, final_pos, obstacle_coords, station_nodes


def build_dual_cache(graph, stations_dict):
    cache_dist = {}
    cache_hops = {}
    for sid, start_v in stations_dict.items():
        dist = {v: float('inf') for v in graph}
        hops = {v: float('inf') for v in graph}
        dist[start_v] = 0.0
        hops[start_v] = 0.0
        
        pq = [(0.0, start_v)]
        while pq:
            d, curr = heapq.heappop(pq)
            if d > dist[curr]:
                continue
            for nxt, weight in graph[curr]:
                new_d = d + weight
                if new_d < dist[nxt]:
                    dist[nxt] = new_d
                    heapq.heappush(pq, (new_d, nxt))
                    
        q = [start_v]
        visited = {start_v}
        while q:
            curr = q.pop(0)
            for nxt, _ in graph[curr]:
                if nxt not in visited:
                    visited.add(nxt)
                    hops[nxt] = hops[curr] + 1.0
                    q.append(nxt)
                    
        for v in graph:
            cache_dist[(v, sid)] = dist[v]
            cache_hops[(v, sid)] = hops[v]
            
    return cache_dist, cache_hops


def make_cost_matrix_local(active_rids, active_vertices, station_ids, cache_dist, working_targets, lookahead_weight):
    c_subset = np.zeros((len(active_rids), len(station_ids)))
    for i, rid in enumerate(active_rids):
        current_v = active_vertices[rid]
        target_v = working_targets[rid]
        for j, sid in enumerate(station_ids):
            dist_to_station = cache_dist.get((current_v, sid), float('inf'))
            dist_to_task = cache_dist.get((target_v, sid), float('inf'))
            c_subset[i, j] = dist_to_station + (lookahead_weight * dist_to_task)
    return c_subset


def build_subset_problem(model, active_rids, station_ids, vertices, cache_dist, soc, caps, load, previous, min_reachable, chargers_per_station, gamma_frac, dev_mult, dev_base, working_targets, lookahead_weight):
    if not active_rids:
        return None, 0

    active_soc = np.array([soc[r] for r in active_rids])
    active_vertices = {r: vertices[r] for r in active_rids}
    
    c_subset = make_cost_matrix_local(active_rids, active_vertices, station_ids, cache_dist, working_targets, lookahead_weight)
    d_subset = dev_mult * c_subset + dev_base 
    
    # ATUALIZAÇÃO: Para impedir o CRASH da simulação, damos um orçamento matemático infinito
    # para a filtragem inicial. O custo letal será cobrado na função evaluate do ACO!
    infinite_budget = np.full(len(active_rids), np.inf)
    true_L = (active_soc / 0.020) - 1.0 # O orçamento físico real
    
    problem = ChargingAssignmentProblem(
        robot_ids=active_rids,
        station_ids=station_ids,
        nominal_cost=c_subset,
        deviation=d_subset,
        travel_budget=infinite_budget, # Sem crashes!
        residual_capacity=caps - load,
        current_load=load,
        total_capacity=caps,
        previous_assignment=previous,
        gamma=gamma_frac * len(active_rids),
        congestion_weight=25.0,
        switching_weight=10.0,
        epsilon=0.05
    )
    problem.charger_capacity = np.full(len(station_ids), chargers_per_station)
    problem.charging_time = np.full(len(station_ids), 5.0) 
    problem.soc = active_soc
    problem.true_budget = true_L
    
    return problem, len(station_ids)


def get_next_step(rid, current_v, state, assignment, graph, cache_dist, stations, rng, working_targets, pos, blocked, station_centers, dynamic_edges, cols):
    def valid_edge(u, v):
        if v in blocked: 
            return False
        r1, c1 = u // cols, u % cols
        r2, c2 = v // cols, v % cols
        if abs(r1 - r2) == 1 and abs(c1 - c2) == 1:
            cross1 = r1 * cols + c2
            cross2 = r2 * cols + c1
            if (cross1, cross2) in dynamic_edges or (cross2, cross1) in dynamic_edges:
                return False
        return True

    if state[rid] == PLUGGED_IN:
        sid = assignment.get(rid)
        if not sid: return current_v
        station_v = stations[sid]
        if current_v == station_v: return current_v
        choices = [u for u, _ in graph[current_v] if valid_edge(current_v, u) and (u not in station_centers or u == station_v)]
        if not choices: return current_v
        return min(choices, key=lambda n: cache_dist.get((n, sid), float('inf')))

    elif state[rid] == WAITING:
        sid = assignment.get(rid)
        if current_v not in station_centers: return current_v
        choices = [u for u, _ in graph[current_v] if valid_edge(current_v, u) and u not in station_centers]
        if not choices: return current_v
        if sid: return min(choices, key=lambda n: cache_dist.get((n, sid), float('inf')))
        return rng.choice(choices)

    elif state[rid] == HEADING:
        sid = assignment.get(rid)
        if not sid: return current_v
        choices = [u for u, _ in graph[current_v] if valid_edge(current_v, u) and u not in station_centers]
        if current_v not in station_centers: choices.append(current_v)
        if not choices: return current_v
        return min(choices, key=lambda n: cache_dist.get((n, sid), float('inf')))
        
    elif state[rid] == WORKING:
        target_v = working_targets[rid]
        if current_v == target_v:
            if current_v in station_centers:
                free = [u for u, _ in graph[current_v] if valid_edge(current_v, u) and u not in station_centers]
                return rng.choice(free) if free else current_v
            return current_v
            
        pq = [(0.0, current_v)]
        came_from = {current_v: None}
        g_score = {current_v: 0.0}
        tx, ty = pos[target_v]
        
        while pq:
            _, curr = heapq.heappop(pq)
            if curr == target_v: break
            for nxt, weight in graph[curr]:
                if not valid_edge(curr, nxt) or nxt in station_centers: continue
                new_g = g_score[curr] + weight
                if nxt not in g_score or new_g < g_score[nxt]:
                    g_score[nxt] = new_g
                    nx, ny = pos[nxt]
                    heuristic = ((nx - tx)**2 + (ny - ty)**2)**0.5
                    heapq.heappush(pq, (new_g + heuristic, nxt))
                    came_from[nxt] = curr
                    
        step = target_v
        if step not in came_from:
            if current_v in station_centers:
                free_neighbors = [u for u, _ in graph[current_v] if valid_edge(current_v, u) and u not in station_centers]
                return rng.choice(free_neighbors) if free_neighbors else current_v
            return current_v 
        while came_from[step] != current_v: step = came_from[step]
        return step


def interpolate_positions(vertices, target, positions, fraction):
    f = fraction * fraction * (3 - 2 * fraction)
    out = {}
    for rid, u in vertices.items():
        v = target[rid]
        x0, y0 = positions[u]
        x1, y1 = positions[v]
        out[rid] = ((1 - f) * x0 + f * x1, (1 - f) * y0 + f * y1)
    return out


def plot_academic_results(macro_metrics_df, aco_snapshots, pheromone_snapshots, internal_pheromone_snapshot, internal_target_epoch, trip_records, out_dir):
    plt.rcParams.update({
        'font.family': 'serif',
        'axes.titlesize': 12,
        'axes.labelsize': 11,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'figure.dpi': 300
    })

    plots_dir = out_dir / 'academic_plots'
    plots_dir.mkdir(exist_ok=True)

    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    ax1.plot(macro_metrics_df['epoch'], macro_metrics_df['fitness_per_agent'], 'k-', linewidth=2, label='Avg Fitness per Agent')
    ax1.plot(macro_metrics_df['epoch'], macro_metrics_df['robust_cost_per_agent'], 'r--', linewidth=1.5, label='Avg Robust Cost')
    ax1.plot(macro_metrics_df['epoch'], macro_metrics_df['congestion_per_agent'], 'b-.', linewidth=1.5, label='Avg Congestion Penalty')
    ax1.set_xlabel('Decision Epoch ($t_k$)')
    ax1.set_ylabel('Normalized Cost / Active Agent')
    ax1.set_title('Macro-Convergence: Normalized D-ACO Cost over Time')
    ax1.grid(True, linestyle=':', alpha=0.7)
    ax1.legend(loc='upper right')
    fig.tight_layout()
    fig.savefig(plots_dir / '1_macro_convergence.pdf')
    plt.close(fig)

    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    ax1.set_xlabel('Decision Epoch ($t_k$)')
    ax1.set_ylabel('Swarm SoC (%)', color='black')
    
    ln1 = ax1.plot(macro_metrics_df['epoch'], macro_metrics_df['avg_soc'] * 100, color='tab:green', linewidth=2, label='Average Swarm SoC')
    ln2 = ax1.plot(macro_metrics_df['epoch'], macro_metrics_df['min_soc'] * 100, color='tab:orange', linewidth=1.5, linestyle='-.', label='Minimum Swarm SoC')
    
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.set_ylim(-5, 105)
    ax1.grid(True, linestyle=':', alpha=0.7)
    
    ax2 = ax1.twinx()  
    ln3 = ax2.step(macro_metrics_df['epoch'], macro_metrics_df['active_agents'], color='tab:purple', linestyle='--', linewidth=1.5, label='Active D-ACO Agents')
    ax2.set_ylabel('Active D-ACO Agents', color='tab:purple')  
    ax2.tick_params(axis='y', labelcolor='tab:purple')
    ax2.set_ylim(0, max(macro_metrics_df['active_agents']) + 2)
    
    lns = ln1 + ln2 + ln3
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
    
    plt.title('Swarm Energetic Health and Workload')
    fig.tight_layout()
    fig.savefig(plots_dir / '2_swarm_health.pdf')
    fig.savefig(plots_dir / '2_swarm_health.png')
    plt.close(fig)

    if aco_snapshots:
        all_curves = [data['fitness_iter'] for data in aco_snapshots.values() if len(data['fitness_iter']) > 0]
        if all_curves:
            fig, ax = plt.subplots(figsize=(7, 4.5))
            max_len = max(len(curve) for curve in all_curves)
            padded_curves = []
            for curve in all_curves:
                padded = list(curve)
                if len(padded) > 0:
                    last_val = padded[-1]
                    while len(padded) < max_len:
                        padded.append(last_val)
                    padded_curves.append(padded)
            matrix = np.array(padded_curves)
            mean_curve = np.mean(matrix, axis=0)
            std_curve = np.std(matrix, axis=0)
            iterations = np.arange(1, max_len + 1)
            ax.plot(iterations, mean_curve, color='#1f77b4', linewidth=2, label='Mean Normalized Fitness')
            ax.fill_between(iterations, np.maximum(0, mean_curve - std_curve), mean_curve + std_curve, color='#1f77b4', alpha=0.2, label='±1 Standard Deviation')
            ax.set_xlabel('ACO Iteration (Internal)')
            ax.set_ylabel('Normalized Best Fitness / Agent')
            ax.set_title('Internal D-ACO Convergence Profile (Statistical Aggregate)')
            ax.grid(True, linestyle=':', alpha=0.7)
            ax.legend(loc='upper right')
            fig.tight_layout()
            fig.savefig(plots_dir / '3_micro_aco_convergence_stat.pdf')
            plt.close(fig)

    if pheromone_snapshots:
        epochs = sorted(list(pheromone_snapshots.keys()))
        all_keys = set()
        for k in epochs:
            all_keys.update(pheromone_snapshots[k].keys())
        if all_keys:
            sorted_keys = sorted(list(all_keys), key=lambda x: (x[1], x[0]))
            P = np.zeros((len(sorted_keys), len(epochs)))
            for j, k in enumerate(epochs):
                robot_totals = {}
                for key in sorted_keys:
                    rid = key[0]
                    robot_totals[rid] = robot_totals.get(rid, 0.0) + pheromone_snapshots[k].get(key, 0.0)
                for i, key in enumerate(sorted_keys):
                    rid = key[0]
                    if robot_totals[rid] > 1e-9:
                        P[i, j] = pheromone_snapshots[k].get(key, 0.0) / robot_totals[rid]
                    else:
                        P[i, j] = 0.0
            
            fig, ax = plt.subplots(figsize=(8.5, 6))
            cax = ax.imshow(P, aspect='auto', cmap='magma', origin='lower', vmin=0.0, vmax=1.0)
            ax.set_xlabel('Decision Epoch ($t_k$)')
            ax.set_title('Pheromone Matrix Evolution (Selection Probability)')
            ax.set_yticks(np.arange(len(sorted_keys)))
            ax.set_yticklabels([key[0] for key in sorted_keys], fontsize=7)
            ax.set_ylabel('Robot ID', fontweight='bold')
            
            current_station = sorted_keys[0][1]
            station_ticks = []
            station_labels = []
            block_start = 0
            for i, key in enumerate(sorted_keys):
                if key[1] != current_station:
                    ax.axhline(i - 0.5, color='white', linewidth=0.8, linestyle='--')
                    station_ticks.append((block_start + i - 1) / 2.0)
                    station_labels.append(current_station)
                    current_station = key[1]
                    block_start = i
            station_ticks.append((block_start + len(sorted_keys) - 1) / 2.0)
            station_labels.append(current_station)
            
            ax2 = ax.twinx()
            ax2.set_ylim(ax.get_ylim()) 
            ax2.set_yticks(station_ticks)
            ax2.set_yticklabels(station_labels, rotation=90, va='center', fontweight='bold', fontsize=9)
            ax2.set_ylabel('Target Charging Station', fontweight='bold')
            ax2.tick_params(axis='y', length=0) 
            
            fig.tight_layout(rect=[0, 0, 0.85, 1])
            cbar_ax = fig.add_axes([0.88, 0.12, 0.03, 0.76])
            fig.colorbar(cax, cax=cbar_ax, label=r'Relative Confidence / Probability')
            fig.savefig(plots_dir / '4_pheromone_heatmap_macro.pdf', bbox_inches='tight')
            fig.savefig(plots_dir / '4_pheromone_heatmap_macro.png', bbox_inches='tight')
            plt.close(fig)

    if internal_pheromone_snapshot:
        iterations_count = len(internal_pheromone_snapshot)
        all_keys = set()
        for p_dict in internal_pheromone_snapshot:
            all_keys.update(p_dict.keys())
        if all_keys:
            sorted_keys = sorted(list(all_keys), key=lambda x: (x[1], x[0]))
            P = np.zeros((len(sorted_keys), iterations_count))
            for j, p_dict in enumerate(internal_pheromone_snapshot):
                robot_totals = {}
                for key in sorted_keys:
                    rid = key[0]
                    robot_totals[rid] = robot_totals.get(rid, 0.0) + p_dict.get(key, 0.0)
                for i, key in enumerate(sorted_keys):
                    rid = key[0]
                    if robot_totals[rid] > 1e-9:
                        P[i, j] = p_dict.get(key, 0.0) / robot_totals[rid]
                    else:
                        P[i, j] = 0.0
                        
            fig, ax = plt.subplots(figsize=(8.5, 6))
            cax = ax.imshow(P, aspect='auto', cmap='magma', origin='lower', vmin=0.0, vmax=1.0)
            ax.set_xlabel('ACO Internal Iteration')
            ax.set_title(f'Decision Probability Evolution (Inside Epoch $t_{{{internal_target_epoch}}}$)')
            ax.set_yticks(np.arange(len(sorted_keys)))
            ax.set_yticklabels([key[0] for key in sorted_keys], fontsize=7)
            ax.set_ylabel('Robot ID', fontweight='bold')
            
            current_station = sorted_keys[0][1]
            station_ticks = []
            station_labels = []
            block_start = 0
            for i, key in enumerate(sorted_keys):
                if key[1] != current_station:
                    ax.axhline(i - 0.5, color='white', linewidth=0.8, linestyle='--')
                    station_ticks.append((block_start + i - 1) / 2.0)
                    station_labels.append(current_station)
                    current_station = key[1]
                    block_start = i
            station_ticks.append((block_start + len(sorted_keys) - 1) / 2.0)
            station_labels.append(current_station)
            
            ax2 = ax.twinx()
            ax2.set_ylim(ax.get_ylim()) 
            ax2.set_yticks(station_ticks)
            ax2.set_yticklabels(station_labels, rotation=90, va='center', fontweight='bold', fontsize=9)
            ax2.set_ylabel('Target Charging Station', fontweight='bold')
            ax2.tick_params(axis='y', length=0) 
            
            fig.tight_layout(rect=[0, 0, 0.85, 1])
            cbar_ax = fig.add_axes([0.88, 0.12, 0.03, 0.76])
            fig.colorbar(cax, cax=cbar_ax, label=r'Relative Confidence / Probability')
            fig.savefig(plots_dir / f'5_pheromone_heatmap_micro_t{internal_target_epoch}.pdf', bbox_inches='tight')
            fig.savefig(plots_dir / f'5_pheromone_heatmap_micro_t{internal_target_epoch}.png', bbox_inches='tight')
            plt.close(fig)

    if len(trip_records) > 0:
        trip_df = pd.DataFrame(trip_records).sort_values(by='nominal_hops').reset_index(drop=True)
        fig, ax = plt.subplots(figsize=(8, 5))
        
        ax.fill_between(trip_df.index, trip_df['nominal_hops'], trip_df['robust_hops'], color='#ff9999', alpha=0.3, label='Safety Margin ($\Gamma$)')
        ax.plot(trip_df.index, trip_df['robust_hops'], color='tab:red', linestyle='-', linewidth=2, label='Robust Bound (Bertsimas-Sim)')
        ax.plot(trip_df.index, trip_df['nominal_hops'], color='tab:blue', linestyle='--', linewidth=2, label='Nominal Time Expectation (Perimeter Hops)')
        ax.scatter(trip_df.index, trip_df['real_epochs'], color='black', marker='x', s=25, alpha=0.8, label='Real Tracked Delay', zorder=4)
        
        ax.set_xlabel('Recharge Mission Index (Sorted by Nominal Time)')
        ax.set_ylabel('Mission Duration (Decision Epochs / Time Steps)')
        ax.set_title('Uncertainty Envelope: Expected vs. Realized Mission Costs')
        ax.grid(True, linestyle=':', alpha=0.7)
        ax.legend(loc='upper left')
        
        fig.tight_layout()
        fig.savefig(plots_dir / '6_robust_survival_envelope.pdf')
        fig.savefig(plots_dir / '6_robust_survival_envelope.png')
        plt.close(fig)


def render_fsm(path, epoch, phase, edges, positions, obstacle_coords, stations, xy, soc, state, problem, result, previous, working_targets, label_fontsize=5.6):
    fig, ax = plt.subplots(figsize=(8, 7))
    cmap = plt.get_cmap('tab10')
    
    station_ids = list(stations.keys())
    colors = {sid: cmap(j % 10) for j, sid in enumerate(station_ids)}
    
    for u, v, _ in edges:
        ax.plot([positions[u][0], positions[v][0]], [positions[u][1], positions[v][1]], color='#eeeeee', lw=0.6, zorder=1)
        
    for ox, oy in obstacle_coords:
        ax.add_patch(plt.Rectangle((ox - 0.5, oy - 0.5), 1, 1, color='#dddddd', zorder=2))
        
    changed = set()
    if result and problem:
        changed = {r for r, s in result.assignment.items() if previous and previous.get(r) != s}
        
    for rid, target_v in working_targets.items():
        # if state[rid] == WORKING:
            tx, ty = positions[target_v]
            ax.scatter(tx, ty, s=20, marker='x', color='k', zorder=2, linewidth=1.2)
            ax.text(tx, ty + 0.10, f'{rid}', color='k', fontsize=4.9, ha='center', fontweight='bold', zorder=2)
        
    for rid, coords in xy.items():
        x, y = coords
        robot_soc = soc[rid]
        robot_state = state[rid]
        
        marker_size = 60  
        
        if robot_state == WORKING:
            color = '#999999'
            edgecolor = '#666666'
            linewidth = 0.5
        else:
            sid = result.assignment.get(rid) if result else None
            color = colors[sid] if sid else '#ff0000'
            edgecolor = 'red' if rid in changed else 'black'
            linewidth = 1.8 if rid in changed else 0.5
            
        ax.scatter(x, y, s=marker_size, color=color, alpha=0.9, edgecolor=edgecolor, linewidth=linewidth, zorder=4)
        
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
        ax.scatter(x, y, s=300, marker='s', color=colors[sid], edgecolor='black', linewidth=1, zorder=7)
        cap = problem.total_capacity[j] if problem else "N/A"
        ax.text(
            x, y + 0.35, f'{sid}\n{assigned_counts[sid]}/{cap}',
            ha='center', fontsize=7, fontweight='bold', color=colors[sid]
        )
        
    ax.set_title(f'$t_{{{epoch}}}$ + {phase:.2f}', fontweight='bold')
    
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
    p.add_argument('--robots', type=int, default=15)
    p.add_argument('--stations', type=int, default=3)
    p.add_argument('--rows', type=int, default=8)
    p.add_argument('--cols', type=int, default=8)
    p.add_argument('--obstacle-prob', type=float, default=0.15)
    p.add_argument('--battery-threshold', type=float, default=0.45)
    
    p.add_argument('--gamma-frac', type=float, default=0.80)
    p.add_argument('--dev-mult', type=float, default=1.5)
    p.add_argument('--dev-base', type=float, default=15.0)
    p.add_argument('--lookahead-weight', type=float, default=0.5)
    p.add_argument('--q-pheromone', type=float, default=200.0)
    
    p.add_argument('--epochs', type=int, default=60)
    p.add_argument('--frames-per-epoch', type=int, default=5)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--output', default='fsm_animation')
    p.add_argument('--save-data', action='store_true', default=True)
    
    p.add_argument('--aco-epochs', type=int, default=25)
    p.add_argument('--ants', type=int, default=25)
    p.add_argument('--max-time', type=float, default=0.25)
    p.add_argument('--capacity-margin', type=float, default=1.4)
    p.add_argument('--chargers-per-station', type=int, default=1)
    p.add_argument('--minimum-reachable', type=int, default=2)
    p.add_argument('--fps', type=float, default=2.0)
    p.add_argument('--save-mp4', action='store_true')
    p.add_argument('--ffmpeg', default='ffmpeg')
    
    p.add_argument('--internal-heatmap-epoch', type=int, default=-1)

    a = p.parse_args()
    out = Path(a.output)
    frames = out / 'animation_frames'
    frames.mkdir(parents=True, exist_ok=True)
    
    rng = np.random.default_rng(a.seed)
    
    graph, edges, pos, obstacle_coords, station_nodes = create_custom_grid_graph(
        a.rows, a.cols, a.stations, a.obstacle_prob, a.seed
    )
    graph_nodes = list(graph.keys())
    
    stations = {f'CS-{i+1}': v for i, v in enumerate(station_nodes)}
    station_ids = tuple(stations.keys())
    station_centers = set(stations.values())
    
    cache_dist, cache_hops = build_dual_cache(graph, stations)
    
    station_zone_nodes = set()
    for v in station_centers:
        r_idx, c_idx = v // a.cols, v % a.cols
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                nr, nc = r_idx + dr, c_idx + dc
                if 0 <= nr < a.rows and 0 <= nc < a.cols:
                    nv = nr * a.cols + nc
                    if nv in graph_nodes:
                        station_zone_nodes.add(nv)
                        
    working_node_pool = [n for n in graph_nodes if n not in station_zone_nodes]
    if not working_node_pool: 
        working_node_pool = [n for n in graph_nodes if n not in station_centers]
    
    robot_node_pool = [n for n in graph_nodes if n not in station_nodes]
    
    if a.robots > int(len(robot_node_pool) * 0.60):
        a.robots = int(len(robot_node_pool) * 0.60)
        
    robot_start_nodes = rng.choice(robot_node_pool, size=a.robots, replace=False)
    robot_ids = [f'R{i:02d}' for i in range(a.robots)]
    vertices = {rid: v for rid, v in zip(robot_ids, robot_start_nodes)}
    
    soc = {r: rng.uniform(a.battery_threshold, 1.0) for r in robot_ids}
    state = {r: WORKING for r in robot_ids}
    
    chosen_initial = rng.choice(working_node_pool, size=a.robots, replace=False)
    working_targets = {rid: int(target) for rid, target in zip(robot_ids, chosen_initial)}
    
    base_cap = np.full(a.stations, np.ceil((a.robots / a.stations) * a.capacity_margin))
    caps = base_cap.astype(int)
    
    time_limit = a.max_time if a.max_time > 0 else None
    
    # ATUALIZAÇÃO DO OTIMIZADOR COM NOVA PENALIDADE
    class ResilientDynamicRobustACO(DynamicRobustACO):
        @staticmethod
        def evaluate(problem: ChargingAssignmentProblem, assignment: np.ndarray):
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
            current_soc = np.asarray(getattr(problem, "soc", np.ones(r)), dtype=float)
            true_budget = getattr(problem, "true_budget", problem.travel_budget)
            
            queue_cost = 0.0
            waiting_cost = 0.0
            starvation_penalty = 0.0
            
            for j in range(len(problem.station_ids)):
                members = [i for i in range(r) if int(assignment[i]) == j]
                members.sort(key=lambda i: (float(current_soc[i]), i))
                occupied = int(problem.current_load[j])
                
                for rank, i in enumerate(members, start=1):
                    absolute_position = occupied + rank
                    wait_cycles = max(0, (absolute_position - 1) // max(1, int(chargers[j])))
                    wait = wait_cycles * float(charge_time[j])
                    
                    queue_cost += queue_weight * max(0, absolute_position - int(chargers[j]))
                    waiting_cost += wait_weight * wait
                    
                    total_expected_time = problem.nominal_cost[i, j] + problem.deviation[i, j] + wait
                    if total_expected_time > true_budget[i]:
                        # O D-ACO não crasha. Tenta escolher o mal menor.
                        starvation_penalty += 1000.0 * (total_expected_time - true_budget[i])

            utilization = total_load / problem.total_capacity
            congestion = float(problem.congestion_weight * np.sum(utilization ** 2))
            
            switching_count = 0
            for i, rid in enumerate(problem.robot_ids):
                previous = problem.previous_assignment.get(rid)
                if previous is not None and previous != problem.station_ids[int(assignment[i])]:
                    switching_count += 1
            switching = float(problem.switching_weight * switching_count)
            
            total_fitness = nominal + robust + congestion + queue_cost + waiting_cost + switching + starvation_penalty
            
            components = {
                "nominal": nominal, "robust": robust, "congestion": congestion,
                "queue": float(queue_cost), "waiting": float(waiting_cost),
                "switching": switching, "switch_count": float(switching_count), "starvation": starvation_penalty
            }
            return total_fitness, components

    model = ResilientDynamicRobustACO(
        epoch=a.aco_epochs, pop_size=a.ants, heuristic_power=2.5,
        evaporation=0.2, forgetting=0.15, q_pheromone=a.q_pheromone, max_time=time_limit
    )
    
    previous_assignment = {}
    paths = []
    frame = 0
    
    macro_data = []
    aco_snapshots = {}
    pheromone_snapshots = {}
    internal_pheromone_snapshot = None 
    
    ongoing_trips = {}
    trip_records = []
    starved_robots = set()
    
    for k in range(a.epochs):
        
        for r in robot_ids:
            if state[r] == WORKING and vertices[r] == working_targets[r]:
                occupied_targets = set(working_targets.values())
                available_targets = [n for n in working_node_pool if n not in occupied_targets]
                if available_targets:
                    working_targets[r] = int(rng.choice(available_targets))
                else:
                    working_targets[r] = int(rng.choice(working_node_pool))
                    
        for r in robot_ids:
            if state[r] == WORKING:
                soc[r] -= rng.uniform(0.03, 0.06)
                if soc[r] < a.battery_threshold:
                    state[r] = HEADING
            elif state[r] == HEADING:
                soc[r] -= rng.uniform(0.010, 0.020)
                sid = previous_assignment.get(r)
                if sid is not None:
                    dist_hops = cache_hops.get((vertices[r], sid), float('inf'))
                    if dist_hops <= 1.0:
                        state[r] = WAITING
            elif state[r] == WAITING:
                soc[r] -= rng.uniform(0.005, 0.010)
            
            if soc[r] <= 0.0:
                starved_robots.add(r)
                soc[r] = 0.0

        for sid, station_v in stations.items():
            waiting = [r for r in robot_ids if state[r] == WAITING and previous_assignment.get(r) == sid]
            plugged = [r for r in robot_ids if state[r] == PLUGGED_IN and previous_assignment.get(r) == sid]
            
            free_spots = a.chargers_per_station - len(plugged)
            
            if free_spots > 0 and waiting:
                waiting.sort(key=lambda r: soc[r])
                for r in waiting[:free_spots]:
                    state[r] = PLUGGED_IN
                    plugged.append(r)
                    
                    if r in ongoing_trips:
                        trip = ongoing_trips.pop(r)
                        real_time = k - trip['start_k']
                        trip_records.append({
                            'robot_id': r,
                            'nominal_hops': trip['nominal_hops'],
                            'robust_hops': trip['robust_hops'],
                            'real_epochs': real_time
                        })
            
            for r in plugged:
                soc[r] = min(1.0, soc[r] + 0.15)
                if soc[r] >= 1.0:
                    state[r] = WORKING
                    if r in previous_assignment: del previous_assignment[r]

        active_rids = [r for r in robot_ids if state[r] in (HEADING, WAITING, PLUGGED_IN)]
        load = np.zeros(a.stations, dtype=int)
        
        problem, kused = build_subset_problem(
            model, active_rids, station_ids, vertices, cache_dist,
            soc, caps, load, previous_assignment, a.minimum_reachable, a.chargers_per_station,
            a.gamma_frac, a.dev_mult, a.dev_base, working_targets, a.lookahead_weight
        )
        
        result = None
        if problem:
            result = model.solve(problem, seed=a.seed + k)
            previous_assignment = dict(result.assignment)
            
            for r in active_rids:
                if state[r] in (HEADING, WAITING):
                    assigned_sid = previous_assignment.get(r)
                    if assigned_sid and r not in ongoing_trips:
                        hop_val = cache_hops.get((vertices[r], assigned_sid), float('inf'))
                        if hop_val != float('inf'):
                            real_nominal = max(0.0, hop_val - 1.0)
                            robust_bound = real_nominal + (a.dev_mult * real_nominal) + a.dev_base
                            ongoing_trips[r] = {
                                'start_k': k,
                                'nominal_hops': real_nominal,
                                'robust_hops': robust_bound
                            }
            
            num_active = len(active_rids)
            avg_soc = np.mean(list(soc.values()))
            min_soc = np.min(list(soc.values())) if len(soc) > 0 else 0.0 
            
            macro_data.append({
                'epoch': k,
                'active_agents': num_active,
                'avg_soc': avg_soc,
                'min_soc': min_soc,
                'fitness_per_agent': result.fitness / num_active if num_active else 0,
                'robust_cost_per_agent': result.components.get('robust', 0) / num_active if num_active else 0,
                'congestion_per_agent': result.components.get('congestion', 0) / num_active if num_active else 0,
                'switches': result.components.get('switch_count', 0),
                'cumulative_starvations': len(starved_robots) 
            })
            
            if num_active > 0 and hasattr(model, 'history_iter'):
                raw_fitness_iter = model.history_iter.get('fitness_iter', [])
                norm_fitness_iter = [fit / num_active for fit in raw_fitness_iter]
                aco_snapshots[k] = {
                    'fitness_iter': norm_fitness_iter
                }
                
                if k == a.internal_heatmap_epoch:
                    internal_pheromone_snapshot = list(model.history_iter.get('pheromone_matrix', []))
            
            pheromone_snapshots[k] = dict(model.pheromone)
            
            for r in active_rids:
                if state[r] in (WAITING, PLUGGED_IN):
                    new_sid = previous_assignment.get(r)
                    if new_sid:
                        dist_hops = cache_hops.get((vertices[r], new_sid), float('inf'))
                        if dist_hops > 1.0:
                            state[r] = HEADING
            
        target = {}
        dynamic_obstacles = set()
        dynamic_edges = set()
        
        state_priority = {PLUGGED_IN: 0, WAITING: 1, HEADING: 2, WORKING: 3}
        
        def get_dist(r):
            if state[r] in (HEADING, WAITING) and previous_assignment.get(r):
                return cache_dist.get((vertices[r], previous_assignment.get(r)), 0)
            elif state[r] == WORKING:
                return cache_dist.get((vertices[r], working_targets[r]), 0)
            return 0
            
        prioritized_rids = sorted(robot_ids, key=lambda r: (state_priority[state[r]], soc[r], get_dist(r)))
        
        unplanned_positions = {vertices[r] for r in robot_ids}
        
        for r in prioritized_rids:
            unplanned_positions.remove(vertices[r])
            blocked = dynamic_obstacles.union(unplanned_positions)
            
            nxt_step = get_next_step(
                r, vertices[r], state, previous_assignment,
                graph, cache_dist, stations, rng, working_targets, pos, 
                blocked, station_centers, dynamic_edges, a.cols
            )
            
            target[r] = nxt_step
            dynamic_obstacles.add(nxt_step)
            dynamic_edges.add((vertices[r], nxt_step))
            
        for sub in range(a.frames_per_epoch):
            fraction = sub / a.frames_per_epoch
            xy = interpolate_positions(vertices, target, pos, fraction)
            fp = frames / f'frame_{frame:05d}.png'
            
            render_fsm(
                fp, k, fraction, edges, pos, obstacle_coords, stations, xy, soc, state,
                problem, result, previous_assignment, working_targets
            )
            paths.append(fp)
            frame += 1
            
        vertices = target
        
    print(f'Geração concluída. Guardados {len(paths)} frames em {out}')
    
    if a.save_data:
        data_dir = out / 'results_data'
        data_dir.mkdir(exist_ok=True)
        
        if len(macro_data) > 0:
            print("\n📈 A exportar métricas e gerar gráficos académicos...")
            df_macro = pd.DataFrame(macro_data)
            df_macro.to_csv(data_dir / 'macro_metrics_evolution.csv', index=False)
            plot_academic_results(df_macro, aco_snapshots, pheromone_snapshots, internal_pheromone_snapshot, a.internal_heatmap_epoch, trip_records, out)

        if aco_snapshots:
            print("💾 A exportar CSV da Micro-convergência...")
            micro_records = []
            for tk, data in aco_snapshots.items():
                for it_idx, fit_val in enumerate(data['fitness_iter']):
                    micro_records.append({
                        'epoch_tk': tk,
                        'iteration': it_idx + 1,
                        'normalized_fitness': fit_val
                    })
            pd.DataFrame(micro_records).to_csv(data_dir / 'micro_convergence_evolution.csv', index=False)

        if pheromone_snapshots:
            print("💾 A exportar CSV do Heatmap de Feromônio Macro...")
            pher_records = []
            epochs_list = sorted(list(pheromone_snapshots.keys()))
            for tk in epochs_list:
                for key, val in pheromone_snapshots[tk].items():
                    pher_records.append({
                        'epoch_tk': tk,
                        'robot_id': key[0],
                        'station_id': key[1],
                        'pheromone_val': val
                    })
            pd.DataFrame(pher_records).to_csv(data_dir / 'pheromone_evolution_macro.csv', index=False)
            
        if internal_pheromone_snapshot:
            print(f"💾 A exportar CSV do Heatmap de Feromônio Micro (Época {a.internal_heatmap_epoch})...")
            micro_pher_records = []
            for it_idx, p_dict in enumerate(internal_pheromone_snapshot):
                for key, val in p_dict.items():
                    micro_pher_records.append({
                        'internal_iteration': it_idx + 1,
                        'robot_id': key[0],
                        'station_id': key[1],
                        'pheromone_val': val
                    })
            pd.DataFrame(micro_pher_records).to_csv(data_dir / f'pheromone_evolution_micro_t{a.internal_heatmap_epoch}.csv', index=False)

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
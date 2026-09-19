"""
Animação de Enxame Robótico com Máquina de Estados Finita (FSM).
Inclui obstáculos estáticos impenetráveis e desvio dinâmico de colisões (Prioritized Planning).
REGRAS FÍSICAS RESTRITAS (COLISÃO ZERO E REALOCAÇÃO):
- Bloqueio de Nós e Cruzamentos: Prevenção absoluta de colisões simultâneas.
- Reversão Dinâmica: Se o D-ACO realocar um robô estacionado para outra base, ele regressa a HEADING.
- Zoneamento: Waypoints de trabalho são gerados estritamente fora das zonas de recarga.
- Tamanho visual dos robôs é constante em todos os estados.
"""

from __future__ import annotations
import argparse
import csv
import heapq
import shutil
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Importação do otimizador D-ACO e funções base
from dynamic_robust_aco import DynamicRobustACO, ChargingAssignmentProblem
from animate_dynamic_swarm import robust_contributions, find_ffmpeg

# Definição dos Estados Físicos
WORKING = 0
HEADING = 1
WAITING = 2      
PLUGGED_IN = 3   


# ============================================================================
# GERADORES DE AMBIENTE (MAPA, OBSTÁCULOS E DIJKSTRA)
# ============================================================================

def create_custom_grid_graph(rows, cols, num_stations, obstacle_prob=0.15, seed=42):
    rng = np.random.default_rng(seed)
    is_obstacle = rng.random((rows, cols)) < obstacle_prob
    
    all_coords = [(r, c) for r in range(rows) for c in range(cols)]
    station_idx = rng.choice(len(all_coords), size=num_stations, replace=False)
    station_coords = [all_coords[i] for i in station_idx]
    
    # GARANTIA ESPACIAL: Limpar obstáculos ao redor das estações (Matriz 3x3)
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


def build_cache_dijkstra(graph, stations_dict):
    cache = {}
    for sid, start_v in stations_dict.items():
        dist = {v: float('inf') for v in graph}
        dist[start_v] = 0.0
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
                    
        for v, d in dist.items():
            cache[(v, sid)] = d
    return cache


def make_cost_matrix_local(vertices_dict, station_ids, cache):
    c_subset = np.zeros((len(vertices_dict), len(station_ids)))
    for i, (rid, v) in enumerate(vertices_dict.items()):
        for j, sid in enumerate(station_ids):
            c_subset[i, j] = cache.get((v, sid), float('inf'))
    return c_subset


def ensure_k_station_reachability_local(c_subset, d_subset, proposed_L, k, slack=0.50):
    L = proposed_L.copy()
    req = c_subset + d_subset
    for i in range(len(L)):
        sorted_req = np.sort(req[i])
        if k <= len(sorted_req):
            L[i] = max(L[i], sorted_req[k - 1] + slack)
    return L

# ============================================================================
# LÓGICA DO FSM E OTIMIZAÇÃO
# ============================================================================

def build_subset_problem(model, active_rids, station_ids, vertices, cache, soc, caps, load, previous, min_reachable, chargers_per_station, gamma_frac=0.25):
    if not active_rids:
        return None, 0

    active_soc = np.array([soc[r] for r in active_rids])
    active_vertices = {r: vertices[r] for r in active_rids}
    
    c_subset = make_cost_matrix_local(active_vertices, station_ids, cache)
    d_subset = 0.15 * c_subset + 0.25
    proposed_L = 40.0 * active_soc - 2.0
    first_k = min(max(1, min_reachable), len(station_ids))
    
    last_problem = None
    for k in range(first_k, len(station_ids) + 1):
        L = ensure_k_station_reachability_local(c_subset, d_subset, proposed_L, k, slack=0.50)
        
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
        problem.charger_capacity = np.full(len(station_ids), chargers_per_station)
        last_problem = problem
        
        if model._maximum_b_matching(problem) is not None:
            return problem, k
            
    raise RuntimeError("O subconjunto de robôs não possui rotas viáveis. Tente aumentar a margem de capacidade.")


def get_next_step(rid, current_v, state, assignment, graph, cache, stations, rng, working_targets, pos, blocked, station_centers, dynamic_edges, cols):
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
        
        if current_v == station_v:
            return current_v
            
        choices = [u for u, _ in graph[current_v] if valid_edge(current_v, u) and (u not in station_centers or u == station_v)]
        if not choices:
            return current_v
            
        return min(choices, key=lambda n: cache.get((n, sid), float('inf')))

    elif state[rid] == WAITING:
        sid = assignment.get(rid)
        
        if current_v not in station_centers:
            return current_v
            
        choices = [u for u, _ in graph[current_v] if valid_edge(current_v, u) and u not in station_centers]
        if not choices:
            return current_v
        
        if sid:
            return min(choices, key=lambda n: cache.get((n, sid), float('inf')))
        return rng.choice(choices)

    elif state[rid] == HEADING:
        sid = assignment.get(rid)
        if not sid: return current_v
        
        choices = [u for u, _ in graph[current_v] if valid_edge(current_v, u) and u not in station_centers]
        if current_v not in station_centers:
            choices.append(current_v)
            
        if not choices:
            return current_v
            
        return min(choices, key=lambda n: cache.get((n, sid), float('inf')))
        
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
            if curr == target_v:
                break
            for nxt, weight in graph[curr]:
                if not valid_edge(curr, nxt) or nxt in station_centers:
                    continue
                    
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
            
        while came_from[step] != current_v:
            step = came_from[step]
            
        return step


def interpolate_positions(vertices, target, positions, fraction):
    out = {}
    for rid, u in vertices.items():
        v = target[rid]
        x0, y0 = positions[u]
        x1, y1 = positions[v]
        out[rid] = (
            (1 - fraction) * x0 + fraction * x1,
            (1 - fraction) * y0 + fraction * y1
        )
    return out


def render_fsm(path, epoch, phase, edges, positions, obstacle_coords, stations, xy, soc, state, problem, result, previous, working_targets, label_fontsize=4.6):
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
        
    for rid, coords in xy.items():
        x, y = coords
        robot_soc = soc[rid]
        robot_state = state[rid]
        
        marker_size = 120  
        
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
    p.add_argument('--battery-threshold', type=float, default=0.35)
    p.add_argument('--epochs', type=int, default=60)
    p.add_argument('--frames-per-epoch', type=int, default=5)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--output', default='fsm_animation')
    
    p.add_argument('--aco-epochs', type=int, default=25)
    p.add_argument('--ants', type=int, default=25)
    p.add_argument('--max-time', type=float, default=0.25)
    p.add_argument('--capacity-margin', type=float, default=1.4)
    p.add_argument('--chargers-per-station', type=int, default=1)
    p.add_argument('--minimum-reachable', type=int, default=2)
    p.add_argument('--fps', type=float, default=2.0)
    p.add_argument('--save-mp4', action='store_true')
    p.add_argument('--ffmpeg', default='ffmpeg')

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
    
    # === ZONEAMENTO DE RECARGA (3x3) PARA EXCLUIR DOS WAYPOINTS ===
    station_zone_nodes = set()
    for v in station_centers:
        r, c = v // a.cols, v % a.cols
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < a.rows and 0 <= nc < a.cols:
                    nv = nr * a.cols + nc
                    if nv in graph_nodes:
                        station_zone_nodes.add(nv)
                        
    # Pool de trabalho é apenas os nós que NÃO fazem parte das zonas de recarga
    working_node_pool = [n for n in graph_nodes if n not in station_zone_nodes]
    if not working_node_pool: # Fallback se o mapa for minúsculo
        working_node_pool = [n for n in graph_nodes if n not in station_centers]
    # ===============================================================
    
    cache = build_cache_dijkstra(graph, stations)
    
    robot_node_pool = [n for n in graph_nodes if n not in station_nodes]
    
    if a.robots > int(len(robot_node_pool) * 0.60):
        print(f"\n⚠️ AVISO: Reduzindo a frota para {int(len(robot_node_pool)*0.60)} robôs para evitar superpopulação.\n")
        a.robots = int(len(robot_node_pool) * 0.60)
        
    robot_start_nodes = rng.choice(robot_node_pool, size=a.robots, replace=False)
    robot_ids = [f'R{i:02d}' for i in range(a.robots)]
    vertices = {rid: v for rid, v in zip(robot_ids, robot_start_nodes)}
    
    soc = {r: rng.uniform(a.battery_threshold, 1.0) for r in robot_ids}
    state = {r: WORKING for r in robot_ids}
    
    # Agora os robôs procuram trabalho apenas no working_node_pool
    working_targets = {r: int(rng.choice(working_node_pool)) for r in robot_ids}
    
    base_cap = np.full(a.stations, np.ceil((a.robots / a.stations) * a.capacity_margin))
    caps = base_cap.astype(int)
    
    model = DynamicRobustACO(
        epoch=a.aco_epochs, pop_size=a.ants, heuristic_power=2.5,
        evaporation=0.2, forgetting=0.15, q_pheromone=8, max_time=a.max_time
    )
    
    previous_assignment = {}
    paths = []
    frame = 0
    
    for k in range(a.epochs):
        # 1. Bateria e Entrada na Fila Virtual
        for r in robot_ids:
            if state[r] == WORKING:
                soc[r] -= rng.uniform(0.06, 0.12)
                if soc[r] < a.battery_threshold:
                    state[r] = HEADING
            elif state[r] == HEADING:
                soc[r] -= rng.uniform(0.010, 0.020)
                sid = previous_assignment.get(r)
                if sid is not None:
                    dist = cache.get((vertices[r], sid), float('inf'))
                    if dist <= 1.5:
                        state[r] = WAITING

        # 2. Gestão de Filas de Espera e Recarga
        for sid, station_v in stations.items():
            waiting = [r for r in robot_ids if state[r] == WAITING and previous_assignment.get(r) == sid]
            plugged = [r for r in robot_ids if state[r] == PLUGGED_IN and previous_assignment.get(r) == sid]
            
            free_spots = a.chargers_per_station - len(plugged)
            
            if free_spots > 0 and waiting:
                waiting.sort(key=lambda r: soc[r])
                for r in waiting[:free_spots]:
                    state[r] = PLUGGED_IN
                    plugged.append(r)
            
            for r in plugged:
                soc[r] = min(1.0, soc[r] + 0.15)
                if soc[r] >= 1.0:
                    state[r] = WORKING
                    working_targets[r] = int(rng.choice(working_node_pool))
                    if r in previous_assignment: del previous_assignment[r]

        # 3. Renovação de Waypoints para WORKING
        for r in robot_ids:
            if state[r] == WORKING and vertices[r] == working_targets[r]:
                working_targets[r] = int(rng.choice(working_node_pool))

        # 4. D-ACO
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
            
            for r in active_rids:
                if state[r] in (WAITING, PLUGGED_IN):
                    new_sid = previous_assignment.get(r)
                    if new_sid:
                        dist = cache.get((vertices[r], new_sid), float('inf'))
                        if dist > 1.5:
                            state[r] = HEADING
            
        # 5. PLANEAMENTO PRIORIZADO COM COLISÃO ZERO (Nós + Arestas Cruzadas)
        target = {}
        dynamic_obstacles = set()
        dynamic_edges = set()
        
        state_priority = {PLUGGED_IN: 0, WAITING: 1, HEADING: 2, WORKING: 3}
        prioritized_rids = sorted(robot_ids, key=lambda r: (state_priority[state[r]], r))
        
        unplanned_positions = {vertices[r] for r in robot_ids}
        
        for r in prioritized_rids:
            unplanned_positions.remove(vertices[r])
            blocked = dynamic_obstacles.union(unplanned_positions)
            
            nxt_step = get_next_step(
                r, vertices[r], state, previous_assignment,
                graph, cache, stations, rng, working_targets, pos, 
                blocked, station_centers, dynamic_edges, a.cols
            )
            
            target[r] = nxt_step
            dynamic_obstacles.add(nxt_step)
            dynamic_edges.add((vertices[r], nxt_step))
            
        # 6. Renderização de Frames
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
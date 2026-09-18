"""Dynamic robust charging-assignment animation with feasibility recovery."""
from __future__ import annotations
import argparse, csv, shutil, subprocess
from collections import Counter
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from dynamic_robust_aco import ChargingAssignmentProblem, DynamicRobustACO
from example_charging_swarm import make_cost_matrix, build_cache
from scalable_scenario import (
    create_grid_graph, choose_station_vertices, generate_robots,
    generate_station_capacities, ensure_k_station_reachability,
)


def walk(robots, graph, rng, probability, steps):
    out = {}
    for rid, start in robots.items():
        v = start
        if rng.random() < probability:
            for _ in range(int(rng.integers(1, steps + 1))):
                v = int(rng.choice([u for u, _ in graph[v]]))
        out[rid] = v
    return out


def diagnose_feasibility(problem, epoch):
    mask = problem.feasible_mask
    counts = mask.sum(axis=1)
    print(f"[diagnostic t{epoch}] residual={int(problem.residual_capacity.sum())}, "
          f"feasible stations/robot min={int(counts.min())}, "
          f"mean={float(counts.mean()):.2f}, max={int(counts.max())}")
    unreachable = np.flatnonzero(counts == 0)
    if unreachable.size:
        print("  robots without station:", [problem.robot_ids[i] for i in unreachable])
    reachable = mask.sum(axis=0)
    for j, sid in enumerate(problem.station_ids):
        print(f"  {sid}: reachable demand={int(reachable[j])}, "
              f"residual capacity={int(problem.residual_capacity[j])}")


def build_feasible_problem(model, robot_ids, station_ids, vertices, cache, soc,
                           capacities, load, previous, minimum_reachable=3,
                           gamma_fraction=0.25):
    """Build a synthetic benchmark and verify joint capacitated feasibility.

    The function starts with k robustly reachable stations per robot. If the
    capacitated matching still fails, k is increased until all stations are
    available. It never bypasses the matching test or returns an unsafe arc.
    """
    c = make_cost_matrix(vertices, station_ids, cache)
    d = 0.15 * c + 0.25
    proposed = 40.0 * soc - 2.0
    first_k = min(max(1, minimum_reachable), len(station_ids))
    last_problem = None
    for k in range(first_k, len(station_ids) + 1):
        L = ensure_k_station_reachability(c, d, proposed, k, slack=0.50)
        problem = ChargingAssignmentProblem(
            robot_ids, station_ids, c, d, L, capacities-load, load, capacities,
            previous_assignment=previous, gamma=gamma_fraction*len(robot_ids),
            congestion_weight=0.20, switching_weight=1.50, epsilon=0.05)
        last_problem = problem
        if model._maximum_b_matching(problem) is not None:
            return problem, k
    diagnose_feasibility(last_problem, "build")
    raise RuntimeError(
        "Synthetic epoch remains infeasible even when all stations are "
        "energetically reachable. Increase --capacity-margin or --stations."
    )


def robust_contributions(problem, result):
    assignment = result.solution.astype(int)
    active = problem.deviation[np.arange(len(problem.robot_ids)), assignment]
    order = np.argsort(-active)
    psi = np.zeros_like(active, dtype=float)
    whole = min(int(np.floor(problem.gamma)), len(active))
    psi[order[:whole]] = active[order[:whole]]
    fraction = float(problem.gamma - np.floor(problem.gamma))
    if fraction > 0.0 and whole < len(active):
        psi[order[whole]] = fraction * active[order[whole]]
    return active, psi


def render(path, k, edges, positions, stations, vertices, soc, problem,
           result, previous, jitter, k_reachable, label_mode="all", label_fontsize=4.8):
    fig, ax = plt.subplots(figsize=(8, 7))
    cmap = plt.get_cmap('tab10')
    colors = {sid: cmap(j % 10) for j, sid in enumerate(problem.station_ids)}
    active_dev, psi = robust_contributions(problem, result)
    for u, v, _ in edges:
        ax.plot([positions[u][0], positions[v][0]],
                [positions[u][1], positions[v][1]], color='#dddddd', lw=.45, zorder=1)
    changed = {r for r, s in result.assignment.items() if previous and previous.get(r) != s}
    if label_mode == "all": labelled = set(problem.robot_ids)
    elif label_mode == "critical": labelled = {rid for i,rid in enumerate(problem.robot_ids) if psi[i] > 0 or rid in changed or soc[i] <= .25}
    else: labelled = set()
    for i, rid in enumerate(problem.robot_ids):
        x, y = positions[vertices[rid]]; dx, dy = jitter[rid]; sid = result.assignment[rid]
        ax.scatter(x+dx, y+dy, s=18+75*soc[i], color=colors[sid], alpha=.82,
                   edgecolor='red' if rid in changed else 'black',
                   linewidth=1.4 if rid in changed else .25, zorder=4)
        if rid in labelled:
            ax.annotate(f"{rid}  SoC={100*soc[i]:.0f}%\n"
                        f"d={active_dev[i]:.2f}  psi={psi[i]:.2f}",
                        xy=(x+dx,y+dy), xytext=(4,5), textcoords='offset points',
                        fontsize=label_fontsize, zorder=8, clip_on=False,
                        bbox=dict(facecolor='white', edgecolor='none', alpha=.70, pad=.6))
    assigned = Counter(result.assignment.values())
    for j, (sid, v) in enumerate(stations.items()):
        x, y = positions[v]; total = int(problem.current_load[j] + assigned[sid])
        ax.scatter(x,y,s=220,marker='s',color=colors[sid],edgecolor='black',zorder=7)
        ax.text(x,y+.35,f"{sid}\n{total}/{int(problem.total_capacity[j])}",
                fontsize=7,ha='center',fontweight='bold',color=colors[sid])
    ax.set_title(f"Dynamic robust allocation, decision epoch $t_{{{k}}}$", fontweight='bold')
    ax.text(.01,.01,f"fitness={result.fitness:.2f} | robust={result.components['robust']:.2f} | "
            f"congestion={result.components['congestion']:.2f} | "
            f"switches={int(result.components['switch_count'])} | k={k_reachable}",
            transform=ax.transAxes,fontsize=7.3,
            bbox=dict(facecolor='white',edgecolor='#aaaaaa',boxstyle='round,pad=.2'))
    ax.text(.99,.01,r"d: active deviation; psi: contribution to $\Psi(X,\Gamma)$",
            transform=ax.transAxes,fontsize=6.2,ha='right')
    ax.set_aspect('equal'); ax.axis('off'); fig.tight_layout()
    fig.savefig(path,dpi=160,bbox_inches='tight'); plt.close(fig)


def find_ffmpeg(requested='ffmpeg'):
    if requested != 'ffmpeg':
        p = Path(requested)
        if p.is_file(): return str(p)
        raise RuntimeError(f"FFmpeg executable not found: {requested}")
    system = shutil.which('ffmpeg')
    if system: return system
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:
        raise RuntimeError("FFmpeg not found. Run: python -m pip install imageio-ffmpeg") from exc


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--robots',type=int,default=30); p.add_argument('--stations',type=int,default=5)
    p.add_argument('--rows',type=int,default=10); p.add_argument('--cols',type=int,default=10)
    p.add_argument('--epochs',type=int,default=15); p.add_argument('--seed',type=int,default=42)
    p.add_argument('--move-probability',type=float,default=.70); p.add_argument('--steps',type=int,default=2)
    p.add_argument('--aco-epochs',type=int,default=35); p.add_argument('--ants',type=int,default=30)
    p.add_argument('--max-time',type=float,default=.30); p.add_argument('--frame-duration',type=int,default=650)
    p.add_argument('--capacity-margin',type=float,default=1.60)
    p.add_argument('--minimum-reachable',type=int,default=3)
    p.add_argument('--soc-consumption-min',type=float,default=.003)
    p.add_argument('--soc-consumption-max',type=float,default=.010)
    p.add_argument('--minimum-soc',type=float,default=.15)
    p.add_argument('--charge-increment',type=float,default=.10)
    p.add_argument('--output',default='dynamic_animation')
    p.add_argument('--save-mp4',action='store_true'); p.add_argument('--fps',type=float,default=2.0)
    p.add_argument('--ffmpeg',default='ffmpeg')
    p.add_argument('--label-mode',choices=['all','critical','none'],default='all')
    p.add_argument('--label-fontsize',type=float,default=4.8)
    a=p.parse_args()
    if a.capacity_margin <= 1.0: raise ValueError('--capacity-margin must be greater than 1')
    if a.soc_consumption_min > a.soc_consumption_max: raise ValueError('invalid SoC consumption interval')

    out=Path(a.output); frames=out/'animation_frames'; frames.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(a.seed); graph,edges,pos=create_grid_graph(a.rows,a.cols)
    stations=choose_station_vertices(a.rows,a.cols,a.stations); station_ids=tuple(stations)
    cache,_=build_cache(graph,stations); robot_ids,vertices=generate_robots(a.robots,graph.keys(),a.seed)
    soc=rng.uniform(.28,.55,a.robots)
    caps=generate_station_capacities(a.robots,a.stations,a.capacity_margin)
    jr=np.random.default_rng(a.seed+999); jitter={r:(jr.uniform(-.14,.14),jr.uniform(-.14,.14)) for r in robot_ids}
    model=DynamicRobustACO(epoch=a.aco_epochs,pop_size=a.ants,heuristic_power=2.5,
        evaporation=.20,forgetting=.15,q_pheromone=8.0,max_time=a.max_time)
    previous={}; paths=[]; history=[]
    for k in range(a.epochs):
        if k:
            # Recharge robots that reached the station assigned in the preceding epoch.
            for i,rid in enumerate(robot_ids):
                sid=previous.get(rid)
                if sid is not None and vertices[rid] == stations[sid]:
                    soc[i]=min(1.0,soc[i]+a.charge_increment)
            vertices=walk(vertices,graph,rng,a.move_probability,a.steps)
            soc=np.maximum(soc-rng.uniform(a.soc_consumption_min,a.soc_consumption_max,a.robots),a.minimum_soc)
        load=np.zeros(a.stations,dtype=int)
        problem,k_used=build_feasible_problem(model,robot_ids,station_ids,vertices,cache,soc,caps,load,
                                              previous,a.minimum_reachable)
        if model._maximum_b_matching(problem) is None:
            diagnose_feasibility(problem,k)
            raise RuntimeError(f"Unexpected infeasible instance at t{k}")
        result=model.solve(problem,seed=a.seed+k)
        fp=frames/f'frame_t{k:03d}.png'; render(fp,k,edges,pos,stations,vertices,soc,problem,result,
                                               previous,jitter,k_used,a.label_mode,a.label_fontsize); paths.append(fp)
        history.append(dict(epoch=k,fitness=result.fitness,nominal=result.components['nominal'],
            robust=result.components['robust'],congestion=result.components['congestion'],
            switching=result.components['switching'],switch_count=int(result.components['switch_count']),
            runtime_ms=1000*result.runtime_seconds,mean_soc=float(soc.mean()),minimum_reachable_used=k_used))
        previous=dict(result.assignment)
        print(f"t{k}: fitness={result.fitness:.2f}, switches={history[-1]['switch_count']}, k={k_used}")
    imgs=[Image.open(x).convert('RGB') for x in paths]
    imgs[0].save(out/'dynamic_swarm_animation.gif',save_all=True,append_images=imgs[1:],
                 duration=a.frame_duration,loop=0,optimize=False)
    for im in imgs: im.close()
    if a.save_mp4:
        exe=find_ffmpeg(a.ffmpeg); mp4=out/'dynamic_swarm_animation.mp4'
        cmd=[exe,'-y','-framerate',str(a.fps),'-i',str(frames/'frame_t%03d.png'),
             '-vf','pad=ceil(iw/2)*2:ceil(ih/2)*2','-c:v','libx264','-pix_fmt','yuv420p',
             '-movflags','+faststart',str(mp4)]
        subprocess.run(cmd,check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        print(f"Saved MP4: {mp4}")
    with (out/'dynamic_swarm_history.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=history[0].keys()); w.writeheader(); w.writerows(history)
    print(f"Saved {len(paths)} frames, GIF, and CSV in {out}")

if __name__=='__main__': main()

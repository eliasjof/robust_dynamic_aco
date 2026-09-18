"""Finite charging queues with strict lowest-SoC-first priority."""
import numpy as np

def queue_report(problem, result):
    soc=np.asarray(problem.soc,float); chargers=np.asarray(problem.charger_capacity,int); times=np.asarray(problem.charging_time,float)
    report={}; by_robot={}
    for j,sid in enumerate(problem.station_ids):
        members=[i for i,rid in enumerate(problem.robot_ids) if result.assignment[rid]==sid]
        members.sort(key=lambda i:(soc[i],i))
        rows=[]
        for rank,i in enumerate(members,1):
            pos=int(problem.current_load[j])+rank; cycles=max(0,(pos-1)//max(1,chargers[j])); wait=cycles*times[j]
            item={"robot_id":problem.robot_ids[i],"queue_order":pos,"queue_size":len(members),"waiting_time":float(wait),"soc":float(soc[i])}
            rows.append(item); by_robot[problem.robot_ids[i]]=item
        report[sid]=rows
    return report,by_robot

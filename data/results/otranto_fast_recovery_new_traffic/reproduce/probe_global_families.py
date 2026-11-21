"""Explore fresh generic broad-heading seeds against received AIS only."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys,time
import numpy as np
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE),str(HERE.parents[3])]
from probe_choose import pilot_inputs,digest
from choose_branch_profile import save_report
from world.ais_buffer_maneuver import AISBufferPlanner,_local_transition
from world.ais_fast_recovery import FeedbackFamily,_rollouts,_margins
from world import Fossen3DOFSaildroneDynamics,SaildroneMPC

def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run-dir',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
 obs,grid,blocked,traffic,nominal,prefer,provenance=pilot_inputs(args.run_dir)
 assert _local_transition.cache_info().currsize==0
 started=time.perf_counter();planner=AISBufferPlanner(Fossen3DOFSaildroneDynamics(),SaildroneMPC(),5,45)
 initial,forcing=planner._motion_inputs(obs);step=_local_transition(planner.dynamics.parameters,5)
 families=[FeedbackFamily(obs.boat.heading,float(heading),flap,gain,damping,0)
           for heading in np.arange(0.,360.,30.) for flap in (planner.lower[1],planner.upper[1])
           for gain in (.5,.8,1.2) for damping in (2.5,5.)]
 states,commands=_rollouts(step,initial,forcing,families,planner.lower,planner.upper,planner.slew,5,60)
 margins=_margins(traffic,states,5);order=np.argsort(-margins,kind='stable')
 report={'provenance':provenance,'scope':'Fresh generic full-compass144 seed native batch; notwholechoose ortruth clearances.',
         'seconds':time.perf_counter()-started,'seed_count':len(families),'best_margin_m':float(margins[order[0]]),
         'ranked_seeds':[{'margin_m':float(margins[ix]),'family':asdict(families[ix])} for ix in order[:12]],
         'source_sha256':digest(__file__),'truth_used_for_selection':False,'saved_controls_used':False}
 save_report(args.output,report);print(json.dumps(report,indent=2))
if __name__=='__main__':main()

"""Finite generic two-stage actuator search, using received AIS and native dynamics."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys,time
import numpy as np
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE),str(HERE.parents[3])]
from probe_choose import pilot_inputs,digest
from choose_branch_profile import save_report
from world.ais_buffer_maneuver import AISBufferPlanner,_local_transition
from world.ais_fast_recovery import _margins
from world import Fossen3DOFSaildroneDynamics,SaildroneMPC

def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run-dir',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
 obs,grid,blocked,traffic,nominal,prefer,provenance=pilot_inputs(args.run_dir)
 assert _local_transition.cache_info().currsize==0
 started=time.perf_counter();planner=AISBufferPlanner(Fossen3DOFSaildroneDynamics(),SaildroneMPC(),5,45)
 initial,forcing=planner._motion_inputs(obs)
 families=[(rudder,flap,rudder,flap,300.) for rudder in (-30.,-15.,0.,15.,30.) for flap in (-15.,-5.,0.,5.,15.)]
 families += [(rudder,flap,last_rudder,last_flap,seconds) for rudder in (-30.,-15.,0.,15.,30.)
              for flap in (-15.,-5.,0.,5.,15.) for last_rudder in (-15.,0.,15.) for last_flap in (-15.,0.,15.)
              for seconds in (30.,60.,90.,120.)]
 family=np.asarray(families);count=len(families);nodes=np.broadcast_to(initial[:,None],(9,count)).copy()
 forcing=np.broadcast_to(forcing[:,None],(5,count)).copy();step=_local_transition(planner.dynamics.parameters,5).map(count)
 states=[nodes.T.copy()];commands=[]
 for tick in range(60):
  demand=np.where((tick*5<family[:,4])[:,None],family[:,:2],family[:,2:4])
  previous=np.degrees(nodes[[7,8]].T)
  controls=np.clip(demand,np.maximum(planner.lower,previous-planner.slew),np.minimum(planner.upper,previous+planner.slew))
  nodes=np.asarray(step(nodes,np.radians(controls).T,forcing));states.append(nodes.T.copy());commands.append(controls)
 states=np.asarray(states);margins=_margins(traffic,states,5);order=np.argsort(-margins,kind='stable')
 report={'provenance':provenance,'scope':'Fresh generic925 two-stage actuator native batch; notwholechoose ortruthclearances.',
         'seconds':time.perf_counter()-started,'seed_count':count,'best_margin_m':float(margins[order[0]]),
         'ranked_seeds':[{'margin_m':float(margins[ix]),'rudder_first_deg':families[ix][0],'flap_first_deg':families[ix][1],
            'rudder_second_deg':families[ix][2],'flap_second_deg':families[ix][3],'switch_seconds':families[ix][4]} for ix in order[:10]],
         'source_sha256':digest(__file__),'truth_used_for_selection':False,'saved_controls_used':False}
 save_report(args.output,report);print(json.dumps({'seconds':report['seconds'],'count':count,'bestmargin':report['best_margin_m'],'best':report['ranked_seeds'][0]},indent=2))
if __name__=='__main__':main()

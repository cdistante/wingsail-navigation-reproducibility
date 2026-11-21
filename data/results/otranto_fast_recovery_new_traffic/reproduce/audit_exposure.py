"""Distinguish scheduled AIS faults from the portion actually experienced before a stop."""
from __future__ import annotations
import argparse,hashlib,json,math,csv
from pathlib import Path

STUDY=Path(__file__).resolve().parents[1]

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def overlap(end,start,stop):return max(0.,min(end,stop)-max(0.,start))

def history_exposure(study,audit,delay,windows,lead):
 directory='independent_traffic_prefix_history' if audit['scope']=='live-prefix-only' else 'independent_traffic_history'
 path=study/directory/(audit['case']+'.csv')
 metrics=('truth_with_reserve_m','truth_full_m','projected_with_reserve_m','true_nominal_m','true_hull_m')
 first={key:None for key in metrics}
 a,b=windows[0];c,d=windows[1]
 bounds=[('before_first_lost_delivery',None,a+delay),('during_first_lost_delivery',a+delay,b+delay),
         ('between_lost_delivery_windows',b+delay,c+delay),('during_second_lost_delivery',c+delay,d+delay),
         ('after_second_loss',d+delay,None)]
 minima={name:{key:None for key in metrics} for name,_,_ in bounds}
 with path.open() as stream:
  for row in csv.DictReader(stream):
   scenario=float(row['scenario_seconds']);step=int(row['step']);identifier=int(row['mmsi'])
   for key in metrics:
    value=float(row[key])
    if value<0:
     if first[key] is None:
      first[key]={'action_step':step+1,'mission_interval_s':[step*5.,step*5.+5.],
                  'encounter_interval_s':[scenario,scenario+5.],'contacts':[identifier],'minimum_in_interval_m':value}
     elif first[key]['action_step']==step+1:
      first[key]['contacts'].append(identifier);first[key]['minimum_in_interval_m']=min(first[key]['minimum_in_interval_m'],value)
    for name,start,stop in bounds:
     if (start is None or scenario>=start) and (stop is None or scenario<stop):
      old=minima[name][key];minima[name][key]=value if old is None else min(old,value)
 return {'first_negative_intervals':first,'minimum_margins_by_delivery_phase':minima,
   'delivery_phase_encounter_bounds_s':{name:[start,stop] for name,start,stop in bounds},
   'initial_full_reserve_failure_precedes_first_lost_delivery':first['truth_with_reserve_m'] is not None and first['truth_with_reserve_m']['mission_interval_s'][1]<=a+delay+lead,
   'history_sha256':digest(path),
   'timing_scope':'First five-second command interval whose independently swept0.05s minimum is negative; not an exact buffer-entry timestamp.',
   'causal_scope':'Observed phase associations only. No matched no-dropout counterfactual was run, and reception changes would alter the control trajectory.'}


def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--study',type=Path,default=STUDY)
 args=parser.parse_args();study=args.study.resolve()
 rows=[]
 final=json.loads((study/'independent_traffic_sweep.json').read_text())
 merged={row['case']:row for row in final}
 prefix=study/'independent_traffic_prefix_sweep.json'
 if prefix.exists():
  for row in json.loads(prefix.read_text()):merged.setdefault(row['case'],row)
 for audit in merged.values():
  run=study/'runs'/audit['case'];meta=json.loads((run/'experiment.json').read_text())
  fixture=json.loads(Path(meta['traffic_fixture']).read_text());report=meta['reporting']
  interval=float(report['interval_s']);delay=float(report['delay_s']);lead=40.;windows=report['drop_emission_windows_s']
  end=5.*audit['applied_actions'];effects=[(a+lead+delay,b+lead+delay) for a,b in windows]
  ramps=[]
  for ramp in fixture['velocity_ramps']:
   first=(math.floor(ramp['start_s']/interval)+1)*interval
   while any(a<=first<b for a,b in windows):first+=interval
   received=first+delay+lead
   ramps.append({'mmsi':ramp['mmsi'],'physical_start_mission_s':ramp['start_s']+lead,'physical_end_mission_s':ramp['end_s']+lead,
      'physical_ramp_duration_experienced_s':overlap(end,ramp['start_s']+lead,ramp['end_s']+lead),
      'first_changed_velocity_report_delivery_mission_s':received,'changed_velocity_report_received':end>=received})
  maximum=audit.get('maximum_applied_held_report_age_s')
  rows.append({'case':audit['case'],'status':audit['mission_status'],'measurement_scope':audit['scope'],'checkpoint_only':audit['scope']=='live-prefix-only','applied_duration_s':end,'requested_checkpoint_s':600.,
    'entire600s_checkpoint_experienced':end>=600.,'initial20s_delay_present':bool(audit['applied_actions']),
    'scheduled40s_drop_delivery_effect_windows_mission_s':effects,
    'drop_delivery_effect_duration_experienced_s':[overlap(end,a,b) for a,b in effects],
    'any_drop_delivery_effect_experienced':any(overlap(end,a,b)>0 for a,b in effects),
    'maximum_applied_held_report_age_s':maximum,'full65s_age_peak_experienced':bool(audit.get('longest_fault_peak_experienced')),
    'per_target_ramp_exposure':ramps,**history_exposure(study,audit,delay,windows,lead),'replay_sha256':audit['replay_sha256']})
 result={'cases':rows,'all_cases_final':all(not row['checkpoint_only'] for row in rows),'source_sha256':digest(Path(__file__)),'independent_sweep_sha256':digest(study/'independent_traffic_sweep.json'),
   'scope':'Observed exposure only. A fault in the frozen schedule is not counted as tested if navigation stopped before its delayed delivery effect. Velocity ramps are distinguished from reports revealing those ramps.'}
 (study/'fault_exposure_check.json').write_text(json.dumps(result,indent=2)+'\n')
 for row in rows:print(row['case'],row['applied_duration_s'],row['any_drop_delivery_effect_experienced'],row['full65s_age_peak_experienced'])

if __name__=='__main__':main()

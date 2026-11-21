"""Separate sheltered report monitoring from postlaunch traffic/fault exposure."""
from __future__ import annotations
import argparse, csv, hashlib, json, math
from pathlib import Path

STUDY=Path(__file__).resolve().parents[1]
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def overlap(a,b,c,d):return max(0.,min(b,d)-max(a,c))


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
 wait=audit['gate_wait_seconds']
 with path.open() as stream:
  for row in csv.DictReader(stream):
   scenario=float(row['scenario_seconds']);step=int(row['step']);identifier=int(row['mmsi'])
   for key in metrics:
    value=float(row[key])
    if value<0:
     if first[key] is None:
      first[key]={'action_step':step+1,'navigation_interval_s':[step*5.,step*5.+5.],
                  'since_monitoring_start_interval_s':[wait+step*5.,wait+step*5.+5.],
                  'encounter_interval_s':[scenario,scenario+5.],'contacts':[identifier],'minimum_in_interval_m':value}
     elif first[key]['action_step']==step+1:
      first[key]['contacts'].append(identifier);first[key]['minimum_in_interval_m']=min(first[key]['minimum_in_interval_m'],value)
    for name,start,stop in bounds:
     if (start is None or scenario>=start) and (stop is None or scenario<stop):
      old=minima[name][key];minima[name][key]=value if old is None else min(old,value)
 return {'first_negative_intervals':first,'minimum_margins_by_delivery_phase':minima,
   'delivery_phase_encounter_bounds_s':{name:[start,stop] for name,start,stop in bounds},
   'history_sha256':digest(path),
   'timing_scope':'First five-second launched command interval whose independently swept0.05s minimum is negative; not an exact buffer-entry timestamp.',
   'causal_scope':'Observed phase associations. Delaying launch changes forcing sampled at UTC and the traffic phase; no isolated causal attribution to traffic geometry, weather or a single algorithm follows.'}


def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--study',type=Path,default=STUDY)
 args=parser.parse_args();study=args.study.resolve()
 final=study/'independent_traffic_sweep.json';prefix=study/'independent_traffic_prefix_sweep.json'
 merged={row['case']:row for row in json.loads(final.read_text())} if final.exists() else {}
 if prefix.exists():
  for row in json.loads(prefix.read_text()):merged.setdefault(row['case'],row)
 assert merged,'No independent full or checkpoint receipts are available.'
 gates={r['case']:r for r in json.loads((study/'independent_departure_gate_check.json').read_text())['cases']}
 rows=[]
 for audit in merged.values():
  run=study/'runs'/audit['case'];meta=json.loads((run/'experiment.json').read_text())
  fixture=json.loads(Path(meta['traffic_fixture']).read_text());report=meta['reporting'];gate=gates[audit['case']]
  interval=float(report['interval_s']);delay=float(report['delay_s']);lead=40.;windows=report['drop_emission_windows_s']
  duration=5.*audit['applied_actions'];wait=gate['gate_wait_seconds'];end=wait+duration
  effects=[(a+lead+delay,b+lead+delay) for a,b in windows]
  monitoring_exposure=[overlap(0.,wait,a,b) for a,b in effects]
  navigation_exposure=[overlap(wait,end,a,b) for a,b in effects]
  ramps=[]
  for ramp in fixture['velocity_ramps']:
   first=(math.floor(ramp['start_s']/interval)+1)*interval
   while any(a<=first<b for a,b in windows):first+=interval
   received=first+delay+lead;a=ramp['start_s']+lead;b=ramp['end_s']+lead
   ramps.append({'mmsi':ramp['mmsi'],'physical_start_since_monitoring_s':a,'physical_end_since_monitoring_s':b,
      'physical_ramp_duration_while_sheltered_s':overlap(0.,wait,a,b),'physical_ramp_duration_underway_s':overlap(wait,end,a,b),
      'first_changed_velocity_report_delivery_since_monitoring_s':received,
      'changed_velocity_report_delivered_before_launch':received<=wait})
  rows.append({'case':audit['case'],'status':audit['mission_status'],'measurement_scope':audit['scope'],
    'checkpoint_only':audit['scope']=='live-prefix-only','gate_wait_seconds':wait,'actual_departure_utc':gate['actual_departure_utc'],
    'applied_navigation_duration_s':duration,'total_monitoring_plus_navigation_s':end,'requested_navigation_checkpoint_s':600.,
    'entire600s_navigation_checkpoint_experienced':duration>=600.,
    'fixed40s_drop_delivery_effect_windows_since_monitoring_start_s':effects,
    'drop_delivery_effect_duration_while_sheltered_s':monitoring_exposure,
    'drop_delivery_effect_duration_underway_s':navigation_exposure,
    'any_drop_delivery_effect_experienced_underway':any(value>0 for value in navigation_exposure),
    'maximum_monitored_instantaneous_report_age_s':gate['maximum_instantaneous_monitored_report_age_s'],
    'maximum_sheltered_held_report_age_s':gate['maximum_monitored_held_age_to_next_poll_s'],
    'full65s_peak_monitored_while_sheltered':gate['65s_report_age_peak_experienced_only_while_sheltered'],
    'maximum_applied_held_report_age_s':audit.get('maximum_applied_held_report_age_s'),
    'full65s_age_peak_experienced_underway':bool(audit.get('longest_fault_peak_experienced')),
    'per_target_ramp_exposure':ramps,**history_exposure(study,audit,delay,windows,lead),
    'replay_sha256':audit['replay_sha256'],'gate_log_sha256':gate['gate_log_sha256']})
 result={'cases':rows,'all_cases_final':all(not row['checkpoint_only'] for row in rows),'source_sha256':digest(Path(__file__)),
   'independent_sweep_sha256':digest(final) if final.exists() else None,
   'independent_prefix_sweep_sha256':digest(prefix) if prefix.exists() else None,
   'scope':'Fixed-UTC faults are separated into sheltered monitoring and launched motion. Avoiding outages by waiting is a departure-scheduling result; it does not establish underway65s-dropout tolerance. No sheltered plant-motion or berth-clearance claim is made.'}
 (study/'fault_exposure_check.json').write_text(json.dumps(result,indent=2)+'\n')
 for row in rows:print(row['case'],row['gate_wait_seconds'],row['applied_navigation_duration_s'],row['drop_delivery_effect_duration_underway_s'],row['maximum_sheltered_held_report_age_s'])

if __name__=='__main__':main()

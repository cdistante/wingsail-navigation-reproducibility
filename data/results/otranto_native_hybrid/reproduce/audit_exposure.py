"""Audit visibility, held-age coverage and clearance by the fixed underway phases."""
from __future__ import annotations
import argparse, csv, hashlib, json
from pathlib import Path

STUDY=Path(__file__).resolve().parents[1]
METRICS=('true_hull_m','true_nominal_m','truth_full_m','truth_with_reserve_m','projected_full_m','projected_with_reserve_m')
PHASES=(('before_acquisition',None,420.),('acquired_before_anchor',420.,560.),
        ('anchor_before_lost_delivery',560.,665.),('first_underway_lost_delivery',665.,705.),
        ('between_underway_losses',705.,725.),('second_underway_lost_delivery',725.,765.),
        ('after_losses_through900s',765.,900.),('later_route',900.,None))
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def overlap(a,b,c,d):return max(0.,min(b,d)-max(a,c))


def history_phases(study,audit):
 directory='independent_traffic_prefix_history' if audit['scope']=='live-prefix-only' else 'independent_traffic_history'
 path=study/directory/(audit['case']+'.csv');new_id=audit['underway_target_mmsi']
 minima={name:{group:{metric:None for metric in METRICS} for group in ('all_physical_or_visible_reports','new_underway_contact','original_fleet')} for name,_,_ in PHASES}
 negatives={name:{metric:0 for metric in METRICS} for name,_,_ in PHASES}
 first_negative={metric:None for metric in METRICS};new_first_negative={metric:None for metric in METRICS}
 visible_times=[];unseen_rows=0
 with path.open() as stream:
  for row in csv.DictReader(stream):
   t=float(row['scenario_seconds']);identifier=int(row['mmsi']);step=int(row['step'])
   phase=next(name for name,a,b in PHASES if (a is None or t>=a) and (b is None or t<b))
   new=identifier==new_id
   if new:
    if row['reported_visible']=='True':visible_times.append(t)
    else:unseen_rows+=1;assert row['projected_full_m']==row['projected_with_reserve_m']==''
   for metric in METRICS:
    value=None if row[metric]=='' else float(row[metric])
    if value is None:continue
    for group in ('all_physical_or_visible_reports','new_underway_contact' if new else 'original_fleet'):
     old=minima[phase][group][metric]
     minima[phase][group][metric]=value if old is None else min(old,value)
    if value<0:
     negatives[phase][metric]+=1
     record={'action_step':step+1,'encounter_interval_s':[t,t+5.],'navigation_interval_s':[step*5.,step*5.+5.],
             'mmsi':identifier,'minimum_within_interval_m':value,'truth_allowance_scope':row['truth_allowance_scope']}
     if first_negative[metric] is None:first_negative[metric]=record
     if new and new_first_negative[metric] is None:new_first_negative[metric]=record
 return {'minimum_clearance_by_phase_m':minima,'negative_contact_interval_counts_by_phase':negatives,
   'first_negative_intervals_all_contacts':first_negative,'first_negative_intervals_new_contact':new_first_negative,
   'first_new_contact_visible_history_s':min(visible_times,default=None),'new_contact_unseen_command_intervals':unseen_rows,
   'history_sha256':digest(path),
   'phase_minimum_scope':'Native0.05s relative/radius-chord minima for each5s command. Preacquisition new-contact truth uses an explicit65s-equivalent hypothetical allowance; there is no fabricated report projection.'}


def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--study',type=Path,default=STUDY)
 args=parser.parse_args();study=args.study.resolve()
 final=study/'independent_traffic_sweep.json';prefix=study/'independent_traffic_prefix_sweep.json'
 merged={r['case']:r for r in json.loads(final.read_text())} if final.exists() else {}
 if prefix.exists():
  for r in json.loads(prefix.read_text()):merged.setdefault(r['case'],r)
 assert merged,'No independent native receipts are available.'
 gates={r['case']:r for r in json.loads((study/'independent_departure_gate_check.json').read_text())['cases']}
 config=json.loads((study/'underway_config.json').read_text())
 assert config['underway_anchor_elapsed_s']==560 and config['underway_first_delivery_s']==420
 assert config['underway_drop_emission_windows_s']==[[645,685],[705,745]]
 rows=[]
 for audit in merged.values():
  run=study/'runs'/audit['case'];meta=json.loads((run/'experiment.json').read_text());gate=gates[audit['case']]
  actual_windows=meta['reporting']['drop_emission_windows_s']
  assert actual_windows==[[85,125],[145,185],[645,685],[705,745]]
  wait=gate['gate_wait_seconds'];duration=5.*audit['applied_actions'];launch_encounter=wait-40.;end=launch_encounter+duration
  initial_receipt_windows=[(105.,145.),(165.,205.)];new_receipt_windows=[(665.,705.),(725.,765.)]
  new_exposure=[overlap(launch_encounter,end,a,b) for a,b in new_receipt_windows]
  rows.append({'case':audit['case'],'status':audit['mission_status'],'measurement_scope':audit['scope'],
    'checkpoint_only':audit['scope']=='live-prefix-only','actual_departure_utc':gate['actual_departure_utc'],
    'gate_wait_seconds':wait,'applied_navigation_duration_s':duration,'applied_encounter_start_s':launch_encounter,'applied_encounter_end_s':end,
    'original_drop_delivery_duration_while_sheltered_s':[overlap(-40.,launch_encounter,a,b) for a,b in initial_receipt_windows],
    'underway_drop_delivery_windows_encounter_s':new_receipt_windows,'underway_lost_delivery_duration_experienced_s':new_exposure,
    'both_full40s_underway_lost_delivery_effects_experienced':all(x==40. for x in new_exposure),
    'maximum_applied_held_report_age_s':audit.get('maximum_applied_held_report_age_s'),
    'both_actual65s_underway_peaks_verified':bool(audit.get('underway_both65s_age_peaks_verified')),
    'underway_peak_interval_records':audit.get('underway_report_age_peak_intervals',[]),
    'underway_contact_first_acquisition_s':audit.get('underway_first_acquisition_encounter_elapsed_s'),
    'active_AIS_or_COLREG_steps_420_to900':audit.get('underway_active_ais_or_colreg_steps_420_to_900_s',0),
    'AIS_modes_420_to900':audit.get('underway_modes_420_to_900_s',{}),
    **history_phases(study,audit),'replay_sha256':audit['replay_sha256'],'gate_log_sha256':gate['gate_log_sha256']})
 result={'cases':rows,'all_cases_final':all(not r['checkpoint_only'] for r in rows),
   'all_underway_peak_pairs_experienced':all(r['both_actual65s_underway_peaks_verified'] for r in rows),
   'source_sha256':digest(Path(__file__)),'underway_config_sha256':digest(study/'underway_config.json'),
   'independent_sweep_sha256':digest(final) if final.exists() else None,'independent_prefix_sweep_sha256':digest(prefix) if prefix.exists() else None,
   'phase_encounter_bounds_s':{name:[a,b] for name,a,b in PHASES},
   'scope':'Original sheltered monitoring and new underway AIS losses have fixed original UTC anchors. All physical contacts are reconstructed, while report projections exist only after real acquisition. Scheduled windows alone do not establish actual exposure or active avoidance.'}
 (study/'fault_exposure_check.json').write_text(json.dumps(result,indent=2)+'\n')
 for r in rows:print(r['case'],r['underway_contact_first_acquisition_s'],r['underway_lost_delivery_duration_experienced_s'],r['both_actual65s_underway_peaks_verified'],r['active_AIS_or_COLREG_steps_420_to900'])

if __name__=='__main__':main()

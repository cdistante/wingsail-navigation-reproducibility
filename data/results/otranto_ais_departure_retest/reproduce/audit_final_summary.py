"""Tabulate the independent departure/trajectory acceptance evidence."""
from __future__ import annotations
import csv, hashlib, json
from pathlib import Path

STUDY=Path(__file__).resolve().parents[1]
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
 paths={name:STUDY/name for name in ('independent_traffic_sweep.json','independent_departure_gate_check.json',
         'independent_acceptance_check.json','independent_pairing_check.json','fault_exposure_check.json')}
 audit=json.loads(paths['independent_traffic_sweep.json'].read_text())
 gates={r['case']:r for r in json.loads(paths['independent_departure_gate_check.json'].read_text())['cases']}
 acceptance=json.loads(paths['independent_acceptance_check.json'].read_text())
 exposure=json.loads(paths['fault_exposure_check.json'].read_text())
 assert len(audit)==8 and exposure['all_cases_final']
 rows=[]
 for r in audit:
  g=gates[r['case']];captures=r['goal_captures']
  rows.append({'case':r['case'],'encounter_id':r['encounter_id'],'wind_from_degrees':r['wind_from_degrees'],
    'guidance':r['case'].rsplit('__',1)[1],'actual_departure_utc':g['actual_departure_utc'],
    'gate_wait_seconds':g['gate_wait_seconds'],'navigation_seconds':r['applied_actions']*5,
    'total_monitoring_plus_navigation_seconds':g['gate_wait_seconds']+r['applied_actions']*5,
    'status':r['mission_status'],'captures':len(captures),
    'ordered_captures':' -> '.join(x['phase']+':'+x['waypoint'] for x in captures),
    'gate_decisions':g['gate_attempts'],'maximum_gate_decision_s':g['maximum_gate_decision_s'],
    'complete_choose_attempts':r['AIS_choose_attempts'],'maximum_complete_choose_s':r['maximum_complete_AIS_choose_s'],
    'complete_applied_selection_attempts':r['AIS_applied_decision_attempts'],
    'maximum_complete_applied_selection_s':r['maximum_complete_AIS_applied_decision_s'],
    'truth_full_plus_reserve_m':r['configured_truth_centered_minimum_m'],
    'report_full_plus_reserve_m':r['configured_model_minimum_m'],
    'maximum_sheltered_held_age_s':g['maximum_monitored_held_age_to_next_poll_s'],
    'maximum_underway_held_age_s':r['maximum_applied_held_report_age_s'],
    'startup_tube_maximum_observed_excess_m':r['maximum_first300s_launch_tube_excess_m'],
    'capture_distance_min_m':min((x['distance_metres'] for x in captures),default=None),
    'capture_distance_max_m':max((x['distance_metres'] for x in captures),default=None)})
 with (STUDY/'independent_arm_outcomes.csv').open('w',newline='') as stream:
  writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
 summary={'cases':len(rows),'completed_loops':sum(r['status']=='completed' for r in rows),
   'verified_captures':sum(r['captures'] for r in rows),'applied_commands':sum(r['applied_actions'] for r in audit),
   'native_0p05s_substeps':sum(r['applied_actions']*100 for r in audit),
   'sheltered_gate_decisions':sum(r['gate_decisions'] for r in rows),
   'complete_choose_attempts':sum(r['complete_choose_attempts'] for r in rows),
   'complete_applied_selection_attempts':sum(r['complete_applied_selection_attempts'] for r in rows),
   'unapplied_final_attempts':sum(r['AIS_applied_decision_unapplied_final_attempts'] for r in audit),
   'maximum_gate_decision_s':max(r['maximum_gate_decision_s'] for r in rows),
   'maximum_complete_choose_s':max(r['maximum_complete_choose_s'] for r in rows),
   'maximum_complete_applied_selection_s':max(r['maximum_complete_applied_selection_s'] for r in rows),
   'truth_full_plus_reserve_minimum_m':min(r['truth_full_plus_reserve_m'] for r in rows),
   'report_full_plus_reserve_minimum_m':min(r['report_full_plus_reserve_m'] for r in rows),
   'truth_nominal_minimum_m':min(r['truth_minimum_nominal_m'] for r in audit),
   'truth_hull_minimum_m':min(r['truth_minimum_hull_m'] for r in audit),
   'maximum_pose_reconstruction_error_m':max(r['maximum_pose_reconstruction_error_m'] for r in audit),
   'maximum_runtime_margin_difference_m':max(r['maximum_runtime_margin_difference_m'] for r in audit),
   'all_acceptance_passed':acceptance['all_passed'],
   'failed_acceptance_checks':sorted({k for row in acceptance['cases'] for k,v in row['checks'].items() if not v}),
   'all_fixed_dropouts_occurred_while_sheltered':all(all(x==40 for x in r['drop_delivery_effect_duration_while_sheltered_s']) and not any(r['drop_delivery_effect_duration_underway_s']) for r in exposure['cases']),
   'scope':'Protected staging is assumed outside the open-water simulation. This tests received-AIS departure scheduling and launched clearance; it does not establish berth safety, physical station keeping or underway65s outage tolerance.',
   'evidence_sha256':{name:digest(path) for name,path in paths.items()},
   'summary_source_sha256':digest(Path(__file__))}
 (STUDY/'independent_final_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
 print(json.dumps(summary,indent=2))

if __name__=='__main__':main()

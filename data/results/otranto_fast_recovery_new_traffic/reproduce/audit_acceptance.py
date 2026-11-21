"""Assess all retained new-traffic cases without excluding stops or slow attempts."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path

STUDY=Path(__file__).resolve().parents[1]
ROOT=STUDY.parents[2]
CASES=[f'{encounter}__otranto_{wind:03d}__{method}' for encounter in ('crossing_fan','opposed_crossing')
       for wind in (355,5) for method in ('classical','transformer')]
EXPECTED=[('outbound',x) for x in ('002','003','004','005')]+[('return',x) for x in ('004','003','002','001')]

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--study',type=Path,default=STUDY)
 parser.add_argument('--minimum-duration',type=float,default=600.)
 parser.add_argument('--full-mission',action='store_true')
 parser.add_argument('--checkpoint',action='store_true',help='Read the live600s checkpoint aggregate; closed short cases remain failures.')
 parser.add_argument('--cases',nargs='+',default=CASES)
 args=parser.parse_args();study=args.study.resolve()
 sweep=study/('independent_traffic_prefix_sweep.json' if args.checkpoint else 'independent_traffic_sweep.json')
 independent={r['case']:r for r in json.loads(sweep.read_text())}
 if args.checkpoint and (study/'independent_traffic_sweep.json').exists():
  for row in json.loads((study/'independent_traffic_sweep.json').read_text()):
   if row['applied_actions']*5<=args.minimum_duration:independent[row['case']]=row
 assessment_snapshot=None
 if args.checkpoint:
  used={case:independent[case] for case in args.cases if case in independent}
  content=(json.dumps(used,indent=2,sort_keys=True)+'\n').encode()
  checksum=hashlib.sha256(content).hexdigest()
  assessment_snapshot=study/'independent_checkpoint_inputs/assessment_sources'/(checksum+'.json')
  assessment_snapshot.parent.mkdir(parents=True,exist_ok=True)
  if assessment_snapshot.exists():assert assessment_snapshot.read_bytes()==content
  else:assessment_snapshot.write_bytes(content)
 protocol=json.loads((STUDY/'protocol.json').read_text());rows=[]
 for case in args.cases:
  path=study/'runs'/case
  closed=(path/'execution.json').exists()
  metadata_path=path/('execution.json' if closed else 'experiment.json')
  if case not in independent or not metadata_path.exists() or not(args.checkpoint or closed):
   rows.append({'case':case,'passed':False,'checks':{'execution_or_checkpoint_and_independent_receipt_exist':False},'state':'missing-outcome'})
   continue
  r=independent[case];execution=json.loads(metadata_path.read_text());encounter=execution['encounter_id']
  inputs_match=all(digest(ROOT/name)==value and digest(path/'source_snapshot'/name)==value for name,value in execution['source_sha256'].items())
  fixture=Path(execution['traffic_fixture']);trace=Path(execution['truth_trace'])
  assert digest(fixture)==protocol['encounters'][encounter]['definition_sha256']
  assert digest(trace)==protocol['encounters'][encounter]['truth_trace_sha256']
  measurement=Path(r.get('replay_measurement_path',path/'navigation.jsonl.gz'))
  assert r['replay_sha256']==digest(measurement)
  if args.checkpoint and r.get('checkpoint_inputs_frozen'):
   log_measurement=Path(r['run_log_measurement_path'])
   assert r['run_log_sha256']==digest(log_measurement)
  assert r['truth_fixture_sha256']==digest(fixture) and r['truth_trace_sha256']==digest(trace)
  captures=r['goal_captures'];observed=[(x['phase'],x['waypoint']) for x in captures]
  margin=r['configured_truth_centered_minimum_m'];reported=r['configured_model_minimum_m']
  checks={
   'process_normally_closed_or_live_checkpoint':(args.checkpoint and not closed) or execution.get('return_code',execution.get('exit_code'))==0,
   'frozen_execution_inputs_unchanged':inputs_match and (not closed or (bool(execution.get('source_unchanged')) and not execution.get('changed_inputs'))),
   'all_complete_choose_attempts_below5s':r['all_AIS_choose_attempts_below_5s'],
   'truth_full_uncertainty_and_reserve_positive':margin is not None and margin>0,
   'report_full_uncertainty_and_reserve_positive':reported is not None and reported>0,
   'all_applied_clearance_violation_counts_zero':bool(r['applied_actions']) and all(x==0 for x in r['negative_action_counts'].values()),
   'minimum_requested_duration_recorded':r['applied_actions']*5>=args.minimum_duration,
   'longest65s_report_age_experienced':bool(r.get('longest_fault_peak_experienced')),
   'recorded_capture_order_valid':observed==EXPECTED[:len(observed)],
   'all_recorded_capture_ranges_le50m':all(x['distance_metres']<=50 for x in captures),
  }
  if args.full_mission:
   checks.update(full_loop_completed=r['mission_status']=='completed',all8_ordered50m_captures=observed==EXPECTED)
  rows.append({'case':case,'passed':all(checks.values()),'checks':checks,'status':r['mission_status'],'applied_commands':r['applied_actions'],
    'all_choose_attempts':r['AIS_choose_attempts'],'unapplied_final_attempts':r['AIS_choose_unapplied_final_attempts'],
    'maximum_complete_choose_s':r['maximum_complete_AIS_choose_s'],'truth_full_plus_reserve_m':margin,'report_full_plus_reserve_m':reported,
    'maximum_applied_report_age_s':r.get('maximum_applied_held_report_age_s'),'captures':len(captures),'replay_sha256':r['replay_sha256']})
 result={'all_outcomes_retained':True,'expected_cases':args.cases,'all_passed':all(x['passed'] for x in rows),
  'scope':'Full loop with all8ordered50m captures' if args.full_mission else (f'Live{args.minimum_duration:g}s checkpoint of the full-run attempt; full-loop completion is untested.' if args.checkpoint else f'Applied encounter pilot lasting at least{args.minimum_duration:g}s; full-loop completion is untested.'),
  'clearance_criterion':'Both independent truth/report full uncertainty plus3mreserve strictlypositive; no claim of nondecreasing margins across deliberately different traffic/faults.',
  'timing_criterion':'All complete AIS choose attempts strictly below5s, including unsuccessful unapplied final attempts.',
  'protocol_sha256':digest(STUDY/'protocol.json'),'independent_sweep_sha256_at_assessment':digest(sweep),
  'immutable_assessment_input_path':str(assessment_snapshot) if assessment_snapshot else None,
  'immutable_assessment_input_sha256':digest(assessment_snapshot) if assessment_snapshot else None,
  'complete_short_sweep_sha256_at_assessment':digest(study/'independent_traffic_sweep.json') if args.checkpoint and (study/'independent_traffic_sweep.json').exists() else None,
  'source_sha256':digest(Path(__file__)),'cases':rows}
 (study/('independent_checkpoint_acceptance_check.json' if args.checkpoint else 'independent_acceptance_check.json')).write_text(json.dumps(result,indent=2)+'\n')
 for row in rows:print(row['case'],row['passed'],row.get('maximum_complete_choose_s'),row.get('truth_full_plus_reserve_m'))
 if not result['all_passed']:raise SystemExit('Retained acceptance failures exist; see explicit per-case receipt.')

if __name__=='__main__':main()

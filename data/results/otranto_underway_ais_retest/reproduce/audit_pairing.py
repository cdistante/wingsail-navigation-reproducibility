"""Check shared traffic-prefix physics between classical and Transformer arms."""
from __future__ import annotations
import argparse,gzip,hashlib,json
from pathlib import Path

STUDY=Path(__file__).resolve().parents[1]

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def frames(path,limit=121):
 with gzip.open(path,'rt') as stream:
  next(stream);result=[]
  for line in stream:
   result.append(json.loads(line))
   if len(result)>=limit:break
 return result

def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--study',type=Path,default=STUDY)
 args=parser.parse_args();study=args.study.resolve();rows=[]
 for encounter in ('crossing_fan','opposed_crossing'):
  for wind in (355,5):
   names=[f'{encounter}__otranto_{wind:03d}__{method}' for method in ('classical','transformer')]
   source=[];metadata=[];paths=[]
   for name in names:
    run=study/'runs'/name
    frozen=study/'independent_checkpoint_inputs'/name/'navigation_600s.jsonl.gz'
    path=frozen if frozen.exists() else run/'navigation.jsonl.gz'
    source.append(frames(path));paths.append(path)
    metadata.append(json.loads((run/'experiment.json').read_text()))
   a,b=source;first_difference=None
   for k,(x,y) in enumerate(zip(a,b)):
    def physical(row):
     nav=row['navigation']
     return {'state':row['state'],'observation':row['observation'],'rudder':nav.get('rudder_command_deg'),'flap':nav.get('flap_command_deg')}
    if physical(x)!=physical(y):first_difference=k;break
   conflicts=[];accepted=0
   for frame in b[1:]:
    nav=frame['navigation'];active=nav.get('temporal_recovery_source') in ('learned','physical release')
    accepted+=bool(active)
    if active and (nav.get('ais_colreg_active_contacts',0)>0 or nav.get('ais_mode')!='route'):
     conflicts.append(nav['step'])
   rows.append({'encounter_id':encounter,'wind_from_degrees':wind,'cases':names,'frames_checked':[len(a),len(b)],
     'identical_frame_counts':len(a)==len(b),'first_physical_difference_at_step':first_difference,
     'states_observations_commands_identical':len(a)==len(b) and first_difference is None,
     'frozen_source_input_maps_identical':metadata[0]['source_sha256']==metadata[1]['source_sha256'],
     'accepted_model_or_guard_updates_in_checked_prefix':accepted,'learned_or_guard_under_AIS_priority_steps':conflicts,
     'recordings':{str(path.relative_to(study)):digest(path) for path in paths}})
 result={'scope':'First600s when available; entire shorter stopped trajectories otherwise. Timing fields and method-label headers are deliberately excluded from physical equality.',
    'all_pairs_identical':all(x['states_observations_commands_identical'] for x in rows),
    'no_applied_model_or_guard_during_AIS_priority':all(not x['learned_or_guard_under_AIS_priority_steps'] for x in rows),
    'cases':rows,'source_sha256':digest(Path(__file__))}
 (study/'independent_pairing_check.json').write_text(json.dumps(result,indent=2)+'\n')
 for row in rows:print(row['encounter_id'],row['wind_from_degrees'],row['frames_checked'],row['states_observations_commands_identical'],row['accepted_model_or_guard_updates_in_checked_prefix'])

if __name__=='__main__':main()

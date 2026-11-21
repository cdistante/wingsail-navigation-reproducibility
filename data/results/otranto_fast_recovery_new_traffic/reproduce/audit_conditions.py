"""Verify every new validation input and preserve differences from the older traffic."""
from __future__ import annotations
import argparse
import gzip
import hashlib
import json
from pathlib import Path

STUDY=Path(__file__).resolve().parents[1]
ROOT=STUDY.parents[2]
BASELINE=STUDY.with_name('otranto_weak_wind_ais_fast_recovery')

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def prefix(path,count=121):
    with gzip.open(path,'rt') as f:
        next(f);rows=[]
        for i,line in enumerate(f):
            if i>=count:break
            r=json.loads(line)
            rows.append({k:r[k] for k in ('state','observation')} | {'commands':{k:r['navigation'].get(k) for k in ('rudder_command_deg','flap_command_deg')}})
    return rows

def normalise(command):
    variables={'--acados-build-dir','--output-dir','--output-gpx','--record-replay','--temporal-recovery-model'}
    result=[];i=0
    while i<len(command):
        if command[i] in variables:i+=2;continue
        if command[i]=='--confirm-temporal-release':i+=1;continue
        result.append(command[i]);i+=1
    return result

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--study',type=Path,default=STUDY);args=parser.parse_args();study=args.study.resolve()
    protocol=json.loads((STUDY/'protocol.json').read_text());cases=[];paired={};worlds=[]
    archives=[json.loads((folder/'archive_field_fingerprints.json').read_text())['fields'] for folder in (BASELINE,STUDY)]
    assert archives[0]==archives[1], 'Native archived environmental fields changed.'
    for scenario in protocol['encounters']:
        for wind in (355,5):
            pair=[]
            for method in ('classical','transformer'):
                arm=f'{scenario}__otranto_{wind:03d}__{method}';run=study/'runs'/arm
                execute=json.loads((run/'execution.json').read_text());old=json.loads((BASELINE/'runs'/f'otranto_{wind:03d}__{method}'/'execution.json').read_text())
                assert execute['source_unchanged'] and not execute['changed_inputs'],arm
                assert execute['scenario_id']==scenario and execute['method']==method and execute['wind_from_degrees']==wind
                assert execute['departure_utc']==protocol['departure_utc'] and execute['encounter_utc']==protocol['encounter_utc']
                assert execute['radius_m']==50 and execute['dt_s']==5 and execute['maximum_steps']==17280
                assert execute['reporting']==protocol['reporting']
                assert execute['protocol_sha256']==digest(STUDY/'protocol.json')
                assert execute['traffic_fixture_sha256']==protocol['encounters'][scenario]['definition_sha256']
                assert execute['traffic_truth_trace_sha256']==protocol['encounters'][scenario]['truth_trace_sha256']
                sources=execute['source_sha256']
                for name,h in sources.items():
                    assert digest(ROOT/name)==h and digest(run/'source_snapshot'/name)==h,(arm,name)
                for name,h in old['source_sha256'].items():
                    if name=='config.py' or name.startswith(('current_route/','data/bathymetry/','data/results/otranto_temporal_hold_')):
                        assert sources[name]==h,(arm,name,'fixed input changed')
                    if name.startswith('world/') and name not in {'world/ais_buffer_maneuver.py','world/ais_clearance_refinement.py','world/ais_fast_recovery.py','world/run_legacy_fossen.py'}:
                        assert sources[name]==h,(arm,name,'plant or unrelated guidance changed')
                world={k:v for k,v in sources.items() if k.startswith('world/') or k=='config.py'};worlds.append(world)
                physical=prefix(run/'navigation.jsonl.gz')
                old0=prefix(BASELINE/'runs'/f'otranto_{wind:03d}__{method}'/'navigation.jsonl.gz',1)[0]
                if physical:
                    assert physical[0]['state']==old0['state']
                    for field in ('time','wind','current','wave','boat'):assert physical[0]['observation'][field]==old0['observation'][field],(arm,field)
                pair.append((execute,physical))
                cases.append({'arm':arm,'return_code':execute['return_code'],'source_files_verified':len(sources),'replay_sha256':digest(run/'navigation.jsonl.gz'),
                              'initial_own_state_and_archive_equal_original':bool(physical),'all_current_and_snapshot_inputs_match':True,
                              'traffic_and_faults_intentionally_changed':True,'protocol_sha256':execute['protocol_sha256']})
            assert normalise(pair[0][0]['command'])==normalise(pair[1][0]['command']),(scenario,wind,'paired arguments differ')
            common=min(map(lambda x:len(x[1]),pair))
            assert pair[0][1][:common]==pair[1][1][:common],(scenario,wind,'early paired physical controls differ')
            paired[f'{scenario}__{wind:03d}']={'identical_initial_physical_fixes':common,'maximum_prefix_requested_s':600}
    assert all(x==worlds[0] for x in worlds)
    result={'all_current_and_snapshot_inputs_match':True,'same_archived_fields_route_initial_own_state_checkpoint':True,
            'new_traffic_and_longer_faults_are_explicit_controls':True,'paired_prefixes':paired,'cases':cases,'script_sha256':digest(__file__)}
    (study/'independent_conditions_check.json').write_text(json.dumps(result,indent=2)+'\n')
    print('Verified all eight scenario/wind/guidance input records and paired physical prefixes.')

if __name__=='__main__':main()

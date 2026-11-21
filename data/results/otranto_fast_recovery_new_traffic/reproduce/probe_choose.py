"""Profile one fresh AIS choose using exact initial pilot or diagnostic inputs.

Use --run-dir for a completed/failed pilot's first saved observation and the
runner's actual first-decision guide scalars. --snapshot-profile accepts an
exact call snapshot from profile_run, including its actual preview arrays.
No saved controls or traffic truth are used for selection. This diagnostic
adds branch wrappers; independent unprofiled workers certify the deadline.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys
from time import perf_counter

PROCESS_STARTED = perf_counter()
HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
ROOT = STUDY.parents[2]
sys.path[:0] = [str(HERE), str(ROOT)]
import numpy as np
from world.ais import AISTarget
from world.ais_avoidance import AISPrediction
from world.ais_buffer_maneuver import AISBufferPlanner, _local_transition
from world.fossen import Fossen3DOFSaildroneDynamics
from world.mpc import SaildroneMPC
from world.navigation_replay import _decode_episode_frame, _decode_grid
from world.occupancy import LocalGrid
from choose_branch_profile import ChooseBranchProfiler, save_report


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def observation(data):
    row={'state':data['boat'], 'observation':data, 'ais_course_forecasts':[], 'navigation':{}, 'wind_grid':None}
    return _decode_episode_frame(row).observation


def snapshot_inputs(path, index):
    contents=json.loads(path.read_text()); row=contents['input_snapshots'][index]
    obs=observation(row['observation'])
    geometry=row['grid'];cells=geometry['cells']
    grid=LocalGrid(np.zeros((1,cells,cells)),('chart',),geometry['resolution'],geometry['span'],
                   geometry['origin_latitude'],geometry['origin_longitude'])
    pred=dict(row['traffic'])
    pred['targets']=tuple(AISTarget(**target) for target in pred['targets'])
    for name in ('positions_metres','velocities_mps','hull_radii_metres','clearance_radii_metres',
                 'uncertainty_growth_mps','report_ages_seconds','extrapolation_limited','preview_growth_seconds'):
        if pred[name] is not None:pred[name]=np.asarray(pred[name],dtype=bool if name=='extrapolation_limited' else float)
    traffic=AISPrediction(**pred)
    return obs,grid,np.asarray(row['blocked'],dtype=bool),traffic,row['nominal_course_deg'],row['prefer_starboard'],{
        'input_kind':'exact diagnostic call snapshot', 'input_profile_sha256':digest(path),
        'snapshot_index':index,'original_attempt':row['attempt'],'saved_controls_used':False}


def pilot_inputs(folder):
    meta=folder/('execution.json' if (folder/'execution.json').exists() else 'experiment.json')
    execution=json.loads(meta.read_text())
    protocol=json.loads((STUDY/'protocol.json').read_text())
    assert execution['encounter_id'] in protocol['encounters']
    reporting=protocol['reporting'];safety=protocol['safety']
    replay=folder/'navigation.jsonl.gz'
    with gzip.open(replay,'rt') as stream:
        next(stream); raw=json.loads(next(stream))
        try:second=json.loads(next(stream))
        except StopIteration:second=None
    assert raw['navigation']['step']==0
    obs=_decode_episode_frame(json.loads(json.dumps(raw))).observation
    saved=_decode_grid((second or raw)['grid'])
    if saved is None:raise ValueError('Initial chart mask is absent.')
    count=saved.blocked.shape[0]
    grid=LocalGrid(np.zeros((1,count,count)),('chart',),saved.resolution,count*saved.resolution,
                   saved.origin_latitude,saved.origin_longitude)
    traffic=AISPrediction.from_observation(obs,safety_buffer_metres=safety['nominal_gap_m'],
            uncertainty_metres=safety['initial_uncertainty_m'],uncertainty_growth_mps=safety['linear_age_growth_mps'],
            acceleration_bound_mps2=safety['acceleration_bound_mps2'],clearance_reserve_metres=safety['additional_reserve_m']).for_preview(
            reporting['preview_age_budget_s'],protocol['dt_s'])
    timings=[json.loads(line.split('AIS decision timing:',1)[1]) for line in (folder/'run.log').read_text().splitlines()
             if 'AIS decision timing:' in line]
    if not timings:raise ValueError('Exact first decision scalars require the per-attempt timing log.')
    first=timings[0]
    nominal=first['nominal_course_deg'];prefer=first['prefer_starboard']
    return obs,grid,saved.blocked,traffic,nominal,prefer,{
       'input_kind':'actual pilot initial observation plus first applied chart, or initial chart if no action applied',
       'run_directory':str(folder.resolve()),'run_metadata_sha256':digest(meta),'replay_sha256':digest(replay),
       'run_log_sha256':digest(folder/'run.log'),'scenario_id':execution['encounter_id'],
       'wind_from_degrees':execution['wind_from_degrees'],'used_applied_first_chart':second is not None,
       'grid_qualification':'If no action was applied, the saved initial static chart is used; branch diagnosis is not an exact controller-restart certificate.',
       'guide_source':'actual first AIS decision timing log, not post-arbitration replay heading',
       'saved_controls_used':False,'truth_used_for_selection':False}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    choice=parser.add_mutually_exclusive_group(required=True)
    choice.add_argument('--run-dir',type=Path)
    choice.add_argument('--snapshot-profile',type=Path)
    parser.add_argument('--snapshot-index',type=int,default=0)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise SystemExit(f'Output already exists: {args.output}')
    before={str(p.resolve()):digest(p) for p in sorted((ROOT/'world').rglob('*.py'))}
    before[str((ROOT/'config.py').resolve())]=digest(ROOT/'config.py')
    before[str(Path(__file__).resolve())]=digest(__file__)
    before[str((HERE/'choose_branch_profile.py').resolve())]=digest(HERE/'choose_branch_profile.py')
    inputs=pilot_inputs(args.run_dir) if args.run_dir else snapshot_inputs(args.snapshot_profile,args.snapshot_index)
    obs,grid,blocked,traffic,nominal,prefer,provenance=inputs
    cache=dict(zip(('hits','misses','maxsize','currsize'),_local_transition.cache_info()))
    assert cache['currsize']==0,'Fresh diagnostic must not warm the native transition.'
    preparation=perf_counter()-PROCESS_STARTED
    profiler=ChooseBranchProfiler().install(AISBufferPlanner)
    started=perf_counter(); error=None;result=None
    try:
        planner=AISBufferPlanner(Fossen3DOFSaildroneDynamics(),SaildroneMPC(),5.,45.)
        constructed=perf_counter()-started
        result=planner.choose(obs,grid,blocked,traffic,nominal,prefer)
    except BaseException as caught:
        error={'type':type(caught).__name__,'representation':repr(caught)}
    finally:
        setup_plus_profiled_choose=perf_counter()-started
        profiler.close()
    payload=profiler.report()
    payload.update({'recorded_utc':datetime.now(timezone.utc).isoformat(),'process_id':os.getpid(),
        'provenance':provenance,'cache_before_constructor':cache,'constructor_seconds':locals().get('constructed'),
        'setup_plus_profiled_choose_seconds':setup_plus_profiled_choose,'preparation_seconds':preparation,
        'maneuver':None if result is None else asdict(result),'terminal_exception':error,
        'source_sha256':before,'changed_source_paths':[name for name,value in before.items() if digest(name)!=value],
        'thread_environment':{name:os.environ.get(name) for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')},
        'role':'Read-only engineering branch profile; not independent unprofiled deadline acceptance.'})
    save_report(args.output,payload)
    print(json.dumps({'output':str(args.output),'seconds':setup_plus_profiled_choose,'exception':error,
                      'branches':[r['branches'] for r in profiler.records]},indent=2),flush=True)
    if error:raise SystemExit(1)


if __name__=='__main__':main()

"""One fresh-process measurement of the entire first AISBufferPlanner.choose.

No native/optimizer warm-up or previous control sequence is supplied. All
outcomes, including unsafe returns and deadline failures, are written.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import gzip
from hashlib import sha256
import json
from pathlib import Path
import sys,time,os,platform

PROCESS_START=time.perf_counter()
STUDY=Path(__file__).resolve().parents[1]
ROOT=STUDY.parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
from world.navigation_replay import _decode_episode_frame,_decode_grid
from world.occupancy import LocalGrid
from world.ais_avoidance import AISPrediction
from world.ais_buffer_maneuver import AISBufferPlanner,_local_transition
from world.fossen import Fossen3DOFSaildroneDynamics
from world.mpc import SaildroneMPC


def digest(path):return sha256(Path(path).read_bytes()).hexdigest()

def serial(value):
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,np.ndarray):return value.tolist()
    raise TypeError(type(value).__name__)

def save(path,value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2,allow_nan=False,default=serial)+'\n');temp.replace(path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',required=True,choices=[f'{g}__otranto_{w:03d}__classical' for g in ('crossing_fan','opposed_crossing') for w in (355,5)])
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--ready-dir',type=Path)
    args=parser.parse_args();args.output.parent.mkdir(parents=True,exist_ok=True)
    source_before={str(p.relative_to(ROOT)):digest(p) for p in (ROOT/'world').rglob('*.py')}
    source_before['config.py']=digest(ROOT/'config.py')
    for path in [STUDY/'protocol.json',STUDY/'reproduce/traffic_fixture.py',*sorted((STUDY/'fixtures').glob('*/ground_truth.json'))]:
        source_before[str(path.relative_to(ROOT))]=digest(path)
    input_run=STUDY/'pilot_studies/c03_initial_step/runs'/args.case
    experiment=json.loads((input_run/'experiment.json').read_text())
    wind=int(experiment['wind_from_degrees']);encounter=experiment['encounter_id']
    replay=input_run/'navigation.jsonl.gz'
    prefix='AIS decision timing: '
    timing=json.loads(next(line.removeprefix(prefix) for line in (input_run/'run.log').read_text().splitlines() if line.startswith(prefix)))
    nominal_course=timing['nominal_course_deg'];prefer_starboard=timing['prefer_starboard']
    with gzip.open(replay,'rt') as f:header=json.loads(next(f));raw=json.loads(next(f))
    frame=_decode_episode_frame(raw);saved=_decode_grid(raw['grid']);n=saved.blocked.shape[0]
    grid=LocalGrid(np.zeros((1,n,n)),('chart',),saved.resolution,n*saved.resolution,saved.origin_latitude,saved.origin_longitude)
    traffic=AISPrediction.from_observation(frame.observation,acceleration_bound_mps2=.026,clearance_reserve_metres=3.).for_preview(65.,5.)
    caches={'local_transition':dict(zip(('hits','misses','maxsize','currsize'),_local_transition.cache_info()))}
    result={'case':args.case,'encounter_id':encounter,'wind_from_degrees':wind,'process_id':os.getpid(),'host':{'system':platform.system(),'machine':platform.machine(),'logical_cpus':os.cpu_count()},
            'source_sha256':source_before,'worker_source_sha256':digest(__file__),'cache_before':caches,
            'launch_only_observation_input':True,'recorded_controls_used':False,'truth_used':False,
            'input_replay_sha256':digest(replay),'input_experiment_sha256':digest(input_run/'experiment.json'),
            'nominal_course_deg':nominal_course,'prefer_starboard':prefer_starboard,
            'input_scope':'Only frozen initial observation/state/chart and prechoose nominal course/preference are read. Previously selected controls/profiles/timings are never supplied to the planner.',
            'thread_environment':{name:os.environ.get(name) for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')},
            'deadline_seconds':5.0,
            'scope':'Entire first AISBufferPlanner.choose including lazy native/fast kernel construction, candidate search and dense/chart/full-preview validation. Constructor time is also reported and included in conservative setup-plus-choose deadline.',
            'import_and_input_preparation_seconds':time.perf_counter()-PROCESS_START}
    assert caches['local_transition']['currsize']==0,'Native kernel was warmed before the decision.'
    if args.ready_dir:
        args.ready_dir.mkdir(parents=True,exist_ok=True)
        (args.ready_dir/(args.output.stem+'.ready')).write_text('ready\n')
        deadline=time.monotonic()+120
        while not (args.ready_dir/'GO').exists():
            if time.monotonic()>deadline:raise RuntimeError('Cold-wave release barrier timed out.')
            time.sleep(.005)
    total_start=time.perf_counter()
    planner=AISBufferPlanner(Fossen3DOFSaildroneDynamics(),SaildroneMPC(),5.,45.)
    result['constructor_seconds']=time.perf_counter()-total_start
    choose_start=time.perf_counter()
    try:
        maneuver=planner.choose(frame.observation,grid,saved.blocked,traffic,nominal_course,prefer_starboard)
        result['choose_seconds']=time.perf_counter()-choose_start
        result['setup_plus_choose_seconds']=time.perf_counter()-total_start
        result['status']='returned_full_clearance_maneuver' if maneuver is not None and maneuver.predicted_margin_metres>0 else 'no_full_clearance_maneuver'
        result['returned_full_clearance_maneuver']=maneuver is not None and maneuver.predicted_margin_metres>0
        result['search_profile']=getattr(planner,'recovery_search_profile',{})
        result['maneuver']=None if maneuver is None else asdict(maneuver)
        result['generated_policy']=None if planner.policy is None else asdict(planner.policy)
    except Exception as error:
        result.update(status='exception',choose_seconds=time.perf_counter()-choose_start,
                      setup_plus_choose_seconds=time.perf_counter()-total_start,error=repr(error))
    result['deadline_met']=result['setup_plus_choose_seconds']<5.
    result['changed_source_paths']=[name for name,h in source_before.items() if digest(ROOT/name)!=h]
    result['inputs_unchanged']=not result['changed_source_paths']
    save(args.output,result)
    print(json.dumps({k:result[k] for k in ('wind_from_degrees','status','choose_seconds','setup_plus_choose_seconds','deadline_met','inputs_unchanged')}),flush=True)


if __name__=='__main__':main()

"""Retain isolated and synchronized four-condition fresh cold decisions, without5s censoring."""
from __future__ import annotations
import argparse,json,os,subprocess,sys,time
from pathlib import Path
from hashlib import sha256

STUDY=Path(__file__).resolve().parents[1]
WORKER=Path(__file__).with_name('cold_choose_worker.py')
CASES=[f'{g}__otranto_{w:03d}__classical' for g in ('crossing_fan','opposed_crossing') for w in (355,5)]

def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--candidate',required=True)
 parser.add_argument('--isolated-replicates',type=int,default=3)
 parser.add_argument('--waves',type=int,default=3)
 parser.add_argument('--timeout',type=float,default=300.)
 args=parser.parse_args()
 assert args.isolated_replicates>0 and args.waves>0 and args.timeout>5
 output=STUDY/'cold_benchmarks'/args.candidate;output.mkdir(parents=True,exist_ok=False)
 records=[];environment=os.environ.copy()
 names=('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')
 for name in names:environment[name]='1'
 def execute(cases,barrier=None):
  running=[]
  for trial,case in cases:
   path=output/(trial+'.json');log=output/(trial+'.log')
   command=[sys.executable,'-B',str(WORKER),'--case',case,'--output',str(path)]
   if barrier:command+=['--ready-dir',str(barrier)]
   stream=log.open('w');process=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,env=environment)
   running.append((trial,case,path,process,stream,time.monotonic()))
  if barrier:
   end=time.monotonic()+args.timeout
   while len(list(barrier.glob('*.ready')))<len(cases):
    if any(p.poll() is not None for _,_,_,p,_,_ in running) or time.monotonic()>end:break
    time.sleep(.01)
   (barrier/'GO').write_text('release\n')
  for trial,case,path,process,stream,start in running:
   remaining=max(.01,args.timeout-(time.monotonic()-start))
   try:process.wait(timeout=remaining);timed_out=False
   except subprocess.TimeoutExpired:process.kill();process.wait();timed_out=True
   stream.close()
   if path.exists():record=json.loads(path.read_text())
   else:record={'case':case,'status':'process_timeout' if timed_out else 'process_error','exit_code':process.returncode,'deadline_met':False,'elapsed_until_failure_s':time.monotonic()-start}
   record['trial']=trial;records.append(record)
   print(trial,record['status'],record.get('setup_plus_choose_seconds'),record['deadline_met'],flush=True)
 for rep in range(args.isolated_replicates):
  for case in CASES:execute([(f'isolated_r{rep+1}_{case}',case)])
 for wave in range(args.waves):
  barrier=output/f'wave{wave+1}_barrier';barrier.mkdir()
  execute([(f'loaded_w{wave+1}_{i}_{case}',case) for i,case in enumerate(CASES)],barrier)
 hashes=[r['source_sha256'] for r in records if 'source_sha256' in r]
 expected=len(CASES)*(args.isolated_replicates+args.waves)
 summary={'candidate':args.candidate,'protocol':vars(args),'all_outcomes_retained':True,'deadline_censoring':False,
  'watchdog_timeout_s':args.timeout,'deadline_seconds':5.,'expected_records':expected,'observed_records':len(records),
  'expected_trials_per_condition':{'isolated':args.isolated_replicates,'loaded':args.waves},
  'observed_trials_per_condition':{case:{mode:sum(r['case']==case and r['trial'].startswith(mode) for r in records) for mode in ('isolated','loaded')} for case in CASES},
  'thread_environment':{name:environment[name] for name in names},
  'all_source_fingerprints_match':len(hashes)==expected and all(x==hashes[0] for x in hashes),
  'all_trials_under5s':len(records)==expected and all(r['deadline_met'] and r.get('inputs_unchanged') and r['status'] not in ('exception','process_timeout','process_error') for r in records),
  'all_trials_return_full_clearance_maneuver':len(records)==expected and all(r.get('returned_full_clearance_maneuver',False) for r in records),
  'clearance_acceptance_scope':'Cold timing and returned maneuver forecast positivity are separate outputs. Actual600s/full-loop native0.05s truth/report envelope clearance must be independently verified; no margin monotonicity is claimed against different prior traffic.',
  'driver_source_sha256':sha256(Path(__file__).read_bytes()).hexdigest(),'records':records}
 (output/'benchmark.json').write_text(json.dumps(summary,indent=2)+'\n')
 print('All complete cold decisions below5s:',summary['all_trials_under5s'],flush=True)
 print('All return full-clearance maneuver:',summary['all_trials_return_full_clearance_maneuver'],flush=True)

if __name__=='__main__':main()

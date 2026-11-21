"""Start complete native sweeps as executions close; serialize aggregate writes."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import time

import audit_independent as independent

STUDY=Path(__file__).resolve().parents[1]
CASES=[f'{encounter}__otranto_{wind:03d}__{method}'
       for encounter in ('crossing_fan','opposed_crossing') for wind in (355,5)
       for method in ('classical','transformer')]


def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--study',type=Path,default=STUDY)
 parser.add_argument('--jobs',type=int,default=4)
 parser.add_argument('--poll-seconds',type=float,default=5.)
 args=parser.parse_args();study=args.study.resolve()
 assert args.jobs>0 and args.poll_seconds>=1.
 history=study/'independent_traffic_history';history.mkdir(exist_ok=True)
 receipts=study/'independent_traffic_receipts';receipts.mkdir(exist_ok=True)
 aggregate=study/'independent_traffic_sweep.json'
 source=Path(independent.__file__);source_hash=independent.digest(source)
 snapshot=study/'audit_source_snapshots'/(source_hash+'.py');snapshot.parent.mkdir(exist_ok=True)
 if snapshot.exists():assert snapshot.read_bytes()==source.read_bytes()
 else:snapshot.write_bytes(source.read_bytes())
 results={r['case']:r for r in json.loads(aggregate.read_text())} if aggregate.exists() else {}
 assert not results,'This rolling full-sweep launcher requires a fresh aggregate; use the standalone --reuse auditor otherwise.'
 submitted=set();pending={};failures=[]
 print(f'Waiting for {len(CASES)} closed executions; up to {args.jobs} native audit workers.',flush=True)
 with ProcessPoolExecutor(max_workers=args.jobs) as pool:
  while len(results)+len(failures)<len(CASES):
   for case in CASES:
    run=study/'runs'/case
    if case in submitted or not(run/'execution.json').exists():continue
    meta=json.loads((run/'experiment.json').read_text())
    task=(str(run),meta['traffic_fixture'],meta['truth_trace'],str(history),None)
    pending[pool.submit(independent.audit,task)]=case;submitted.add(case)
    print('Started full native audit:',case,flush=True)
   for future,case in list(pending.items()):
    if not future.done():continue
    pending.pop(future)
    try:
     result=future.result()
     assert result['audit_source_sha256']==source_hash
     independent.save_json(receipts/(case+'.json'),result)
     results[case]=result
     independent.save_json(aggregate,[results[name] for name in sorted(results)])
     print(f"Published {case}: {result['mission_status']}, {len(result['goal_captures'])}/8; "
           f"truth/report full+3={result['configured_truth_centered_minimum_m']}/"
           f"{result['configured_model_minimum_m']}m",flush=True)
    except Exception as error:
     failure={'case':case,'audit_exception':repr(error),'audit_source_sha256':source_hash}
     failures.append(failure)
     independent.save_json(study/'independent_full_audit_failures.json',failures)
     print('Independent audit exception retained:',failure,flush=True)
   if len(results)+len(failures)<len(CASES):time.sleep(args.poll_seconds)
 if failures:raise SystemExit('Some complete native audits failed; retained explicit exception receipts.')
 print('All8 complete native sweeps published.',flush=True)

if __name__=='__main__':main()

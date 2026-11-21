"""Read-only readiness of live600s audit prefixes; never stop or alter a run."""
from __future__ import annotations
import argparse
import gzip
import json
from pathlib import Path

STUDY=Path(__file__).resolve().parents[1]


def inspect(folder,prefix_steps):
    path=folder/'navigation.jsonl.gz';closed=(folder/'execution.json').exists()
    steps=[];error=None
    if path.exists():
        try:
            with gzip.open(path,'rt') as stream:
                next(stream)
                for line in stream:
                    try:row=json.loads(line)
                    except json.JSONDecodeError:break
                    steps.append(row['navigation']['step'])
                    if len(steps)>=prefix_steps+1:break
        except (EOFError,OSError,StopIteration) as exc:
            error=type(exc).__name__
    available=len(steps)>=prefix_steps+1
    if steps:assert steps==list(range(len(steps))),(folder.name,steps[-1])
    return {'arm':folder.name,'closed':closed,'complete_prefix_available':available,
            'ready_for_independent_audit':closed or available,
            'applied_prefix_steps_available':max(0,len(steps)-1),'requested_prefix_steps':prefix_steps,
            'insufficient_exposure_if_closed':closed and not available,'live_read_note':error}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',type=Path,default=STUDY)
    parser.add_argument('--prefix-seconds',type=int,default=600)
    args=parser.parse_args()
    if args.prefix_seconds<=0 or args.prefix_seconds%5:parser.error('Use positive5s-aligned prefix seconds.')
    study=args.study.resolve();prefix_steps=args.prefix_seconds//5
    folders=sorted((study/'runs').iterdir()) if (study/'runs').exists() else []
    rows=[inspect(folder,prefix_steps) for folder in folders if folder.is_dir() and (folder/'experiment.json').exists()]
    report={'scope':'Readiness only: independent audit has not been substituted by this reader; no run is killed or modified.',
            'requested_prefix_seconds':args.prefix_seconds,'cases':rows}
    output=study/'live_checkpoint_readiness.json'
    temp=output.with_suffix('.tmp');temp.write_text(json.dumps(report,indent=2)+'\n');temp.replace(output)
    for row in rows:print(row['arm'],row['ready_for_independent_audit'],row['applied_prefix_steps_available'],'closed',row['closed'])

if __name__=='__main__':main()

"""Verify frozen recordings against encoded MP4 receipts and save decoded QA stills."""
from __future__ import annotations
import argparse
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

STUDY=Path(__file__).resolve().parents[1]
ROOT=STUDY.parents[2]
sys.path.insert(0,str(ROOT))
from world.navigation_replay import _ffmpeg_path

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',type=Path,default=STUDY)
    args=parser.parse_args();study=args.study.resolve()
    result=json.loads((study/'comparison.json').read_text())
    output=study/'media_verification'/'decoded';output.mkdir(parents=True,exist_ok=True)
    rows={}
    for arm,metrics in result['runs'].items():
        folder=study/'runs'/arm;video=folder/'navigation.mp4'
        receipt=json.loads((folder/'video_verification.json').read_text())
        assert digest(video)==receipt['video_sha256'] and digest(folder/'navigation.jsonl.gz')==receipt['replay_sha256']
        assert digest(folder/'track.gpx')==receipt['gpx_sha256']
        assert digest(study/'reproduce/render_round_trip.py')==receipt['render_script_sha256']
        with gzip.open(folder/'navigation.jsonl.gz','rt') as stream:
            next(stream);frames=[json.loads(line) for line in stream]
        indices=receipt['source_frame_indices'];present=set(indices)
        launch_summary_frames=int(receipt['launch_summary_frames'])
        assert launch_summary_frames>0 and indices[:launch_summary_frames]==[0]*launch_summary_frames
        assert len(frames)==receipt['source_frame_count']
        assert indices[0]==0 and indices[-1]==len(frames)-1 and indices==sorted(indices)
        assert set(range(min(181,len(frames)))).issubset(present)
        colreg={f['navigation']['step'] for f in frames if f['navigation'].get('ais_colreg_active_contacts',0)>0}
        assert colreg.issubset(present)
        for capture in metrics['captures']:
            assert {k for k in (capture['step']-1,capture['step'],capture['step']+1) if k<len(frames)}.issubset(present)
        worst=metrics['independent_clearance']['minimum_details']['truth_with_reserve']['action_step']
        picks={0,launch_summary_frames,indices.index(worst),len(indices)-1}
        picks.add(indices.index(len(frames)-1))
        for recovery in (False,True):
            stern=[i for i,step in enumerate(indices) if frames[step]['state'].get('surge_speed') is not None
                   and frames[step]['state']['surge_speed']<-.05
                   and bool(frames[step]['navigation'].get('ais_recovery_active',0))==recovery]
            if stern:picks.add(min(stern,key=lambda i:frames[indices[i]]['state']['surge_speed']))
        confirmed=next((i for i,step in enumerate(indices) if frames[step]['navigation'].get('temporal_recovery_source')=='physical release'),None)
        if confirmed is not None:picks.add(confirmed)
        chosen=sorted(picks)
        pattern=output/f'{arm}_decoded_%03d.png'
        selection='select='+'+'.join(f'not(n-{i})' for i in chosen)
        command=[str(_ffmpeg_path()),'-v','error','-i',str(video),'-vf',selection,'-fps_mode','passthrough','-y',str(pattern)]
        decoded=subprocess.run(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
        if decoded.returncode:raise RuntimeError(decoded.stderr)
        stills=[]
        for counter,index in enumerate(chosen,1):
            path=output/f'{arm}_decoded_{counter:03d}.png'
            assert path.is_file() and path.stat().st_size>10000
            stills.append({'source_step':indices[index],'encoded_frame':index,'path':str(path.relative_to(study)),'sha256':digest(path)})
        rows[arm]={'decoded_frames':receipt['decoded_frames'],'source_frames':len(frames),'unique_physical_steps':len(present),
                   'launch_summary_frames':launch_summary_frames,
                   'captures_and_neighbors_retained':len(metrics['captures']),'all_COLREG_updates_retained':len(colreg),
                   'first900s_all_updates_retained':True,'video_sha256':digest(video),'replay_sha256':receipt['replay_sha256'],
                   'still_frames':stills}
        print(arm,len(chosen),'decoded QA frames',flush=True)
    report={'runs':rows,'selection_scope':'First and final, minimum clearance, worst signed sternway inside and outside AIS recovery, first physical release when present; complete MP4 decode is recorded by renderer.',
            'script_sha256':digest(Path(__file__)),'comparison_sha256':digest(study/'comparison.json')}
    (study/'media_verification.json').write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()

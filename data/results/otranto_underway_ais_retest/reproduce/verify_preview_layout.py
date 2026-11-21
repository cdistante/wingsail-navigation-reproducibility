"""Bind final preview bounds/capture coverage; decoded QA remains independent."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
STUDY=Path(__file__).resolve().parents[1]
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reviewed-png',type=Path,action='append',default=[])
    args=parser.parse_args();renderer=digest(STUDY/'reproduce/render_round_trip.py');runs={};total=0;captures=0
    receipts=sorted((STUDY/'runs').glob('*/video_verification.json'))
    if len(receipts)!=8:raise ValueError('Need all8 closed renderer receipts')
    for receipt in receipts:
        video=json.loads(receipt.read_text());arm=receipt.parent.name;layout=STUDY/'media_preview'/f'{arm}_layout.json'
        checks=json.loads(layout.read_text());seen=set();insets=0
        assert video['render_script_sha256']==renderer and video['decoded_frames']==len(video['source_frame_indices'])
        for row in checks:
            x,y,w,h=row['text_bbox'];lx,ly,lw,lh=row['legend_bbox'];cx,cy,cw,ch=row['canvas_bbox']
            assert x>=cx and x+w<=cx+cw and y+h<=cy+ch and y>=ly+lh+6
            assert row['render_script_sha256']==renderer and row['preview_replay_sha256']==video['replay_sha256']
            assert row['preview_scope']=='closed native full mission'
            seen.add(row['source_step']);insets+=row['tactical_inset_bbox'] is not None
        for arrival in video['arrivals']:
            step=arrival['step'];assert step in seen
            assert all(i in video['source_frame_indices'] for i in range(max(0,step-1),min(video['source_frame_count'],step+2)))
            captures+=1
        assert set(video['underway_retained_source_steps'])<=set(video['source_frame_indices'])
        total+=len(checks);runs[arm]={'layout_sha256':digest(layout),'checked_frames':len(checks),'checked_tactical_insets':insets,
                                   'capture_fixes_checked':len(video['arrivals']),'all_capture_neighbors_encoded':True,
                                   'all_recorded_underway_window_states_encoded':True,
                                   'preview_png_sha256':{Path(row['png']).name:digest(Path(row['png'])) for row in checks}}
    report={'all_final_previews_passed':True,'all_eight_renderer_receipts_present':True,'scope':'Preview/layout and source-index checks. Independent decoded-video QA is separate.',
            'renderer_sha256':renderer,'playback_index_sha256':digest(STUDY/'playback.md'),'checked_preview_frames':total,'capture_fixes_and_neighbors_verified':captures,'runs':runs,
            'reviewed_preview_png_sha256':{str(path.resolve()):digest(path) for path in args.reviewed_png},'source_sha256':digest(Path(__file__))}
    path=STUDY/'media_layout_verification.json';tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(report,indent=2)+'\n');tmp.replace(path)
    print('Bound',total,'layout checks and',captures,'capture fixes/neighbors.')
if __name__=='__main__':main()

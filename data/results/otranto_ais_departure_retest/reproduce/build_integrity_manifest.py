"""Bind closed numerical/protocol evidence without racing live media/paper exports."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

STUDY=Path(__file__).resolve().parents[1]
EXTENSIONS={'.json','.jsonl','.csv','.txt','.log','.md','.py','.gpx','.gz','.npz','.nc'}
EXCLUDED_NAMES={'audit_outputs_sha256.json','deliverable_verification.json','media_verification.json',
                'media_layout_verification.json','paper_layout_verification.json','paper_package_verification.json',
                'video_verification.json','video_render.log','playback.md'}


def digest(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    result={}
    for path in sorted(STUDY.rglob('*')):
        if not path.is_file():continue
        name=path.relative_to(STUDY)
        if any(part in {'__pycache__','media_verification'} for part in name.parts):continue
        if path.name in EXCLUDED_NAMES or path.suffix in {'.tmp','.pyc','.mp4','.png'}:continue
        if name.parts[0]=='previous_paper' or 'source_snapshot' in name.parts or path.suffix in EXTENSIONS:
            result[name.as_posix()]=digest(path)
    for name,value in result.items():assert digest(STUDY/name)==value,name
    path=STUDY/'audit_outputs_sha256.json'
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(result,indent=2)+'\n');temporary.replace(path)
    print('Verified and bound',len(result),'closed numeric/protocol/source artifacts.')

if __name__=='__main__':main()

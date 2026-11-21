"""Verify the delivered predecessor without editing any predecessor file."""
from __future__ import annotations
import hashlib,json
from pathlib import Path
STUDY=Path(__file__).resolve().parents[1]
def digest(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def main():
    provenance=json.loads((STUDY/'baseline_provenance.json').read_text())
    base=Path(provenance['baseline_study']);path=base/'audit_outputs_sha256.json'
    if digest(path)!=provenance['baseline_manifest_sha256']:raise ValueError('Baseline manifest changed')
    manifest=json.loads(path.read_text());changed=[name for name,value in manifest.items() if digest(base/name)!=value]
    changed.extend(str(Path('runs')/arm/name) for arm,files in provenance['matched_arm_bindings_sha256'].items()
                   for name,value in files.items() if digest(base/'runs'/arm/name)!=value)
    report={'baseline_study':str(base),'manifest_entries_checked':len(manifest),
            'matched_arms_checked':len(provenance['matched_arm_bindings_sha256']),
            'all_preserved':not changed,'changed_files':changed,
            'baseline_manifest_sha256':digest(path),'script_sha256':digest(Path(__file__))}
    target=STUDY/'baseline_preservation_check.json';tmp=target.with_suffix('.tmp')
    tmp.write_text(json.dumps(report,indent=2)+'\n');tmp.replace(target)
    print(json.dumps(report,indent=2))
    if changed:raise SystemExit('Prior delivered evidence changed')
if __name__=='__main__':main()

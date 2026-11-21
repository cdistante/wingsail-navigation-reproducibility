"""Audit newly closed actors using only released native process slots."""
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
ARMS = {f"crossing_fan__otranto_{wind:03d}__{mode}" for wind in (5, 355) for mode in ("physics", "hybrid")}
audited = set()
while audited != ARMS:
    closed = {name for name in ARMS if (STUDY/"runs"/name/"execution.json").exists()}
    if closed-audited:
        jobs = min(2, len(closed))
        command = [sys.executable, "-B", str(HERE/"audit_native_hybrid.py"), "--reuse", "--jobs", str(jobs)]
        for name in sorted(closed): command.extend(("--case", name))
        print("Auditing closed cases: "+", ".join(sorted(closed))+f"; workers={jobs}", flush=True)
        subprocess.run(command, cwd=STUDY.parents[2], check=True)
        audited = closed
    if audited != ARMS: time.sleep(5.)
print("All four native arms independently audited.", flush=True)

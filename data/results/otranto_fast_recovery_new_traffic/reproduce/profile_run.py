"""Run exact navigation arguments with diagnostic AIS branch timing wrappers.

Example: replace run_archive.py in a preserved pilot command with this file,
then add --profile-output /an/unused/profile.json. All other arguments pass to
run_archive unchanged. This is diagnostic, not the unprofiled deadline gate.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
from time import perf_counter
from unittest.mock import patch

PROCESS_STARTED = perf_counter()
HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
ROOT = STUDY.parents[2]
sys.path[:0] = [str(HERE), str(ROOT)]

from choose_branch_profile import ChooseBranchProfiler, save_report


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_hashes(arguments):
    paths = sorted((ROOT/'world').rglob('*.py')) + [ROOT/'config.py', Path(__file__),
             HERE/'choose_branch_profile.py', HERE/'run_archive.py', HERE/'traffic_fixture.py']
    protocol = STUDY/'protocol.json'
    if protocol.exists(): paths.append(protocol)
    for index, value in enumerate(arguments):
        if value == '--traffic-fixture': paths.append(Path(arguments[index + 1]).resolve())
        elif value.startswith('--traffic-fixture='): paths.append(Path(value.split('=',1)[1]).resolve())
    return {str(p.resolve()): digest(p) for p in paths}


def main():
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument('--profile-output', type=Path, required=True)
    options, runner_args = parser.parse_known_args()
    if options.profile_output.exists():
        raise SystemExit(f'Profile output already exists: {options.profile_output}')
    before = source_hashes(runner_args)
    from world.ais_buffer_maneuver import AISBufferPlanner
    from run_archive import main as run_archive_main

    profiler = ChooseBranchProfiler().install(AISBufferPlanner)
    error = None
    preparation = perf_counter() - PROCESS_STARTED
    try:
        with patch.object(sys, 'argv', [str(HERE/'run_archive.py'), *runner_args]):
            run_archive_main()
    except BaseException as caught:
        error = {'type': type(caught).__name__, 'representation': repr(caught)}
        raise
    finally:
        profiler.close()
        payload = profiler.report()
        payload.update({
            'command': [sys.executable, '-B', str(Path(__file__)), '--profile-output', str(options.profile_output), *runner_args],
            'runner_arguments': runner_args,
            'process_id': os.getpid(), 'recorded_utc': datetime.now(timezone.utc).isoformat(),
            'source_sha256': before,
            'changed_source_paths': [name for name, value in before.items() if digest(name) != value],
            'preparation_before_navigation_seconds': preparation,
            'whole_process_seconds': perf_counter() - PROCESS_STARTED,
            'terminal_exception': error,
            'role': 'Read-only engineering diagnosis, not independent acceptance timing.',
        })
        save_report(options.profile_output, payload)
        print('Saved AIS branch diagnostic:', options.profile_output, flush=True)


if __name__ == '__main__':
    main()

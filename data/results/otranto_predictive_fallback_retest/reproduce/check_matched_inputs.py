"""Confirm that retesting changes the AIS search while preserving the scenario inputs."""
from pathlib import Path
import gzip
import hashlib
import json

STUDY = Path(__file__).resolve().parents[1]
ROOT = STUDY.parents[2]
BASE = STUDY.with_name('otranto_fast_recovery_new_traffic')
CHANGED_WORLD = {'world/run_legacy_fossen.py', 'world/ais_guidance.py', 'world/ais_buffer_maneuver.py'}
NEW_WORLD = 'world/ais_predictive_fallback.py'


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def normalize(command):
    result = command[:]
    # Path changes needed to isolate run output or use copied fixtures.
    for flag in ('--acados-build-dir', '--output-dir', '--output-gpx', '--record-replay', '--traffic-fixture'):
        result[result.index(flag) + 1] = flag
    result[2] = Path(result[2]).name
    if '--ais-predictive-fallback' in result:
        result.remove('--ais-predictive-fallback')
    return result


def initial(path):
    with gzip.open(path, 'rt') as stream:
        return json.loads(next(stream)), json.loads(next(stream))


def main():
    config = json.loads((STUDY / 'retest_config.json').read_text())
    for name, value in config['immutable_base_inputs_sha256'].items():
        assert digest(STUDY / name) == value == digest(BASE / name), name
    matched = {}
    for folder in sorted((STUDY / 'runs').iterdir()):
        old = BASE / 'runs' / folder.name
        reread = json.loads((folder / 'execution.json').read_text())
        baseline = json.loads((old / 'execution.json').read_text())
        assert normalize(reread['command']) == normalize(baseline['command']), folder.name
        assert reread['command'].count('--ais-predictive-fallback') == 1
        assert reread['return_code'] == 0 and reread['source_unchanged'] and not reread['changed_inputs']
        source = reread['source_sha256']
        for name, checksum in source.items():
            assert digest(ROOT / name) == digest(folder / 'source_snapshot' / name) == checksum, name
        for name, checksum in baseline['source_sha256'].items():
            if name == 'config.py' or name.startswith(('current_route/', 'data/bathymetry/', 'data/results/otranto_temporal_hold_')):
                assert source[name] == checksum, name
            if name.startswith('world/') and name not in CHANGED_WORLD:
                assert source[name] == checksum, name
        assert {name for name in source if name.startswith('world/')} - set(baseline['source_sha256']) == {NEW_WORLD}
        actual_header, actual = initial(folder / 'navigation.jsonl.gz')
        base_header, prev = initial(old / 'navigation.jsonl.gz')
        assert actual_header['route'] == base_header['route']
        assert (actual['state'], actual['observation']) == (prev['state'], prev['observation'])
        matched[folder.name] = {'source_files_checked': len(source), 'only_requested_option_differs': True,
                                'initial_state_and_observation_match': True, 'replay_sha256': digest(folder / 'navigation.jsonl.gz')}
    assert len(matched) == 8
    data = {'all_cases_match_original_inputs': True, 'cases': matched, 'script_sha256': digest(Path(__file__))}
    (STUDY / 'matched_inputs_verification.json').write_text(json.dumps(data, indent=2) + '\n')
    print(f'Matched {len(matched)} immutable inputs, initial observations and run configurations.')


if __name__ == '__main__':
    main()

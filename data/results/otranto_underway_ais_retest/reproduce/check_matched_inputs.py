"""Verify allowed regatta/underway changes against the preserved departure study."""
from pathlib import Path
import argparse
from functools import lru_cache
import gzip
import hashlib
import json
import shlex

STUDY = Path(__file__).resolve().parents[1]
ROOT = STUDY.parents[2]
BASE = STUDY.with_name('otranto_ais_departure_retest')
CHANGED_WORLD = {'world/run_legacy_fossen.py'}
NEW_WORLD = 'world/regatta_guidance.py'
CASES = {f'{scenario}__otranto_{wind:03d}__{method}'
         for scenario in ('crossing_fan', 'opposed_crossing')
         for wind in (355, 5) for method in ('classical', 'transformer')}


@lru_cache(maxsize=None)
def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def normalize(command):
    result = command[:]
    for flag in ('--acados-build-dir', '--output-dir', '--output-gpx', '--record-replay', '--traffic-fixture'):
        assert result.count(flag) == 1, flag
        result[result.index(flag) + 1] = flag
    result[2] = Path(result[2]).name
    if '--regatta-time-to-mark' in result:
        assert result.count('--regatta-time-to-mark') == 1
        result.remove('--regatta-time-to-mark')
        assert result.count('--underway-traffic-fixture') == 1
        fixture_pos = result.index('--underway-traffic-fixture')
        del result[fixture_pos:fixture_pos + 2]
        for start, end in (('645', '685'), ('705', '745')):
            pos = result.index(start)
            assert result[pos - 1:pos + 2] == ['--ais-drop-reports', start, end]
            del result[pos - 1:pos + 2]
    return result


def initial(path):
    with gzip.open(path, 'rt') as stream:
        return json.loads(next(stream)), json.loads(next(stream))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prefix', action='store_true',
                        help='Check immutable commands/snapshots and first observations; do not write the final receipt.')
    args = parser.parse_args()
    config = json.loads((STUDY / 'underway_config.json').read_text())
    for name, checksum in config['immutable_base_inputs_sha256'].items():
        assert digest(STUDY / name) == checksum == digest(BASE / name), name
    folders = {folder.name: folder for folder in (STUDY / 'runs').iterdir() if folder.is_dir()}
    assert set(folders) == CASES, 'Missing or unexpected canonical arm'
    matched = {}
    for arm, folder in sorted(folders.items()):
        old = BASE / 'runs' / arm
        metadata_name = 'experiment.json' if args.prefix else 'execution.json'
        metadata_path = folder / metadata_name
        assert metadata_path.is_file(), (arm, 'Final verification requires an execution closure; use --prefix for a live check.')
        reread = json.loads(metadata_path.read_text())
        experiment = json.loads((folder / 'experiment.json').read_text())
        baseline = json.loads((old / 'execution.json').read_text())
        command = reread['command']
        assert command == experiment['command'] == shlex.split((folder / 'command.txt').read_text()), arm
        assert normalize(command) == normalize(baseline['command']), (arm, 'Unexpected command change')
        for flag in ('--ais-departure-gate', '--ais-predictive-fallback', '--regatta-time-to-mark', '--underway-traffic-fixture'):
            assert command.count(flag) == 1, (arm, flag)
        assert command.count('--ais-drop-reports') == 4
        if not args.prefix:
            assert reread['return_code'] == 0 and reread['source_unchanged'] and not reread['changed_inputs'], arm
        source = reread['source_sha256']
        assert source == experiment['source_sha256']
        for name, checksum in source.items():
            assert digest(ROOT / name) == digest(folder / 'source_snapshot' / name) == checksum, (arm, name)
        for name, checksum in config['frozen_runtime_sha256'].items():
            assert source[name] == checksum, (arm, name)
        for name, checksum in baseline['source_sha256'].items():
            if name == 'config.py' or name.startswith(('current_route/', 'data/bathymetry/', 'data/results/otranto_temporal_hold_')):
                assert source[name] == checksum, (arm, name)
            if name.startswith('world/') and name not in CHANGED_WORLD:
                assert source[name] == checksum, (arm, name)
        base_world = {name for name in baseline['source_sha256'] if name.startswith('world/')}
        actual_world = {name for name in source if name.startswith('world/')}
        assert actual_world == base_world | {NEW_WORLD}, arm
        scenario = reread['encounter_id']
        extra = config['underway_encounters'][scenario]
        assert Path(reread['underway_traffic_fixture']) == STUDY / extra['definition']
        assert digest(Path(reread['underway_traffic_fixture'])) == reread['underway_traffic_fixture_sha256'] == extra['definition_sha256']
        assert digest(Path(reread['underway_truth_trace_csv'])) == reread['underway_truth_trace_sha256'] == extra['truth_trace_sha256']
        assert reread['reporting']['drop_emission_windows_s'] == [[85, 125], [145, 185], [645, 685], [705, 745]]
        assert reread['original_reporting'] == baseline['reporting']
        for key, expected in (('underway_first_emission_s', 400), ('underway_first_delivery_s', 420),
                              ('underway_reference_elapsed_s', 560), ('underway_anchor_utc', '2026-10-01T00:00:00Z')):
            assert reread[key] == expected, (arm, key)
        assert reread['underway_drop_emission_windows_s'] == config['underway_drop_emission_windows_s']
        assert reread['underway_frame_retention_original_epoch_s'] == config['underway_frame_retention_original_epoch_s']
        assert reread['retest_config_sha256'] == digest(STUDY / 'underway_config.json')
        assert reread['guidance_objective'] == config['guidance_objective']
        actual_header, actual = initial(folder / 'navigation.jsonl.gz')
        base_header, prev = initial(old / 'navigation.jsonl.gz')
        assert actual_header['route'] == base_header['route'], arm
        gate = json.loads((folder / 'departure_gate.json').read_text())
        assert actual['state'] == prev['state'] == gate['certificate']['own_state'], arm
        assert actual['observation'] == prev['observation'], arm
        assert actual['observation']['ais_targets'] == gate['certificate']['received_reports'], arm
        assert len(actual['observation']['ais_targets']) == 3, 'New contact must not exist in the preacquisition launch reports'
        assert actual['observation']['wind']['direction'] == reread['wind_from_degrees']
        assert reread.get('actual_departure_utc', gate['actual_departure_utc']) == baseline['actual_departure_utc'] == gate['actual_departure_utc']
        assert reread.get('gate_wait_seconds', gate['gate_wait_seconds']) == baseline['gate_wait_seconds'] == gate['gate_wait_seconds']
        assert digest(folder / 'departure_gate.jsonl') == gate['log_sha256']
        matched[arm] = {
            'source_files_checked': len(source),
            'only_underway_and_regatta_options_differ': True,
            'initial_state_and_gate_selected_reports_match': True,
            'initial_marine_forcing_and_route_match_departure_baseline': True,
            'new_contact_absent_before_acquisition': True,
            'closed_execution_verified': not args.prefix,
        }
        if not args.prefix:
            matched[arm]['replay_sha256'] = digest(folder / 'navigation.jsonl.gz')
    data = {'all_cases_match_original_inputs': True, 'cases': matched,
            'scope': 'Live immutable-input and first-observation check only; no final execution or replay hash is asserted.' if args.prefix
                     else 'Closed executions preserve baseline inputs except the explicit underway contact/loss windows and regatta guidance changes.',
            'final_execution_closures_verified': not args.prefix,
            'underway_config_sha256': digest(STUDY / 'underway_config.json'),
            'script_sha256': digest(Path(__file__))}
    if not args.prefix:
        (STUDY / 'matched_inputs_verification.json').write_text(json.dumps(data, indent=2) + '\n')
    print(json.dumps({'cases_checked': len(matched), 'prefix_only': args.prefix,
                      'final_receipt_written': not args.prefix, 'all_checks_passed': True}, indent=2))


if __name__ == '__main__':
    main()

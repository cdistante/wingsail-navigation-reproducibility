"""Bind final underway-traffic navigation, exposure, media and paper evidence."""
from pathlib import Path
import argparse
from functools import lru_cache
import hashlib
import json
import math
import re
import zipfile

STUDY = Path(__file__).resolve().parents[1]
PAPER = STUDY.parents[2] / 'reports/journal_manuscript'
EXPECTED_ORDER = [('outbound', x) for x in ('002', '003', '004', '005')] + [
    ('return', x) for x in ('004', '003', '002', '001')]
CASES = {f'{scenario}__otranto_{wind:03d}__{method}'
         for scenario in ('crossing_fan', 'opposed_crossing')
         for wind in (355, 5) for method in ('classical', 'transformer')}
RECEIPTS = (
    'comparison.json', 'independent_acceptance_check.json', 'independent_departure_gate_check.json',
    'independent_traffic_sweep.json', 'independent_final_summary.json', 'independent_pairing_check.json',
    'fault_exposure_check.json', 'matched_inputs_verification.json', 'baseline_preservation_check.json',
    'audit_outputs_sha256.json', 'video_render.log', 'media_verification.json', 'media_layout_verification.json',
    'paper_layout_verification.json', 'paper_package_verification.json', 'paper_native_compile.json',
)


@lru_cache(maxsize=None)
def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def read(name):
    return json.loads((STUDY / name).read_text())


def extrema(rows, key, maximum=False):
    values = [row[key] for row in rows if row.get(key) is not None]
    return (max if maximum else min)(values) if values else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-ready', action='store_true',
                        help='List missing final artifacts without writing the delivery receipt or building a manifest.')
    args = parser.parse_args()
    required = [STUDY / name for name in RECEIPTS] + [STUDY / 'playback.md',
                PAPER / 'manuscript.pdf', PAPER / 'journal_manuscript_sources.zip']
    for arm in sorted(CASES):
        required.extend(STUDY / 'runs' / arm / name for name in
                        ('execution.json', 'navigation.jsonl.gz', 'navigation.mp4', 'track.gpx',
                         'departure_gate.json', 'departure_gate.jsonl', 'video_verification.json'))
    missing = [str(path) for path in required if not path.is_file()]
    if args.check_ready:
        print(json.dumps({'all_required_paths_exist': not missing, 'missing': missing,
                          'final_receipt_written': False, 'integrity_manifest_built': False}, indent=2))
        return
    assert not missing, ('Final artifacts are still pending', missing)
    comparison = read('comparison.json')['runs']
    acceptance = read('independent_acceptance_check.json')
    accepted = {row['case']: row for row in acceptance['cases']}
    audits = {row['case']: row for row in read('independent_traffic_sweep.json')}
    gates = {row['case']: row for row in read('independent_departure_gate_check.json')['cases']}
    exposure = read('fault_exposure_check.json')
    exposures = {row['case']: row for row in exposure['cases']}
    media = read('media_verification.json')['runs']
    config = read('underway_config.json')
    matched = read('matched_inputs_verification.json')
    for rows in (comparison, accepted, audits, gates, exposures, media, matched['cases']):
        assert set(rows) == CASES, 'Missing or unexpected final arm'
    assert matched['all_cases_match_original_inputs'] and matched['final_execution_closures_verified']
    assert matched['underway_config_sha256'] == digest(STUDY / 'underway_config.json')
    assert read('baseline_preservation_check.json')['all_preserved']
    assert exposure['all_cases_final'] and not any(row['checkpoint_only'] for row in exposures.values())
    assert acceptance['scope'] == 'Full loop with all8ordered50m captures'
    assert all(row['passed'] == all(row['checks'].values()) for row in accepted.values())
    assert acceptance['all_passed'] == all(row['passed'] for row in accepted.values())
    assert acceptance['independent_sweep_sha256_at_assessment'] == digest(STUDY / 'independent_traffic_sweep.json')
    assert acceptance['underway_config_sha256'] == digest(STUDY / 'underway_config.json')
    summary_receipt = read('independent_final_summary.json')
    assert summary_receipt['all_acceptance_passed'] == acceptance['all_passed']
    paired = read('independent_pairing_check.json')
    assert paired['no_applied_model_or_guard_during_AIS_priority']
    completed = 0
    tactical_updates = 0
    for arm, run in comparison.items():
        captures = run['captures']
        audit = audits[arm]
        folder = STUDY / 'runs' / arm
        execution = json.loads((folder / 'execution.json').read_text())
        gate = json.loads((folder / 'departure_gate.json').read_text())
        render = json.loads((folder / 'video_verification.json').read_text())
        actual_order = [(c['phase'], c['waypoint']) for c in captures]
        assert actual_order == EXPECTED_ORDER[:len(actual_order)], arm
        assert all(c['distance_metres'] <= 50 for c in captures), arm
        assert len(captures) == len(audit['goal_captures']), arm
        for reported, independent in zip(captures, audit['goal_captures']):
            assert (reported['phase'], reported['waypoint'], reported['step']) == (
                independent['phase'], independent['waypoint'], independent['step']), arm
            assert math.isclose(reported['distance_metres'], independent['distance_metres'], abs_tol=1e-6), arm
        completed += actual_order == EXPECTED_ORDER and run['status'] == 'completed'
        assert digest(folder / 'navigation.jsonl.gz') == media[arm]['replay_sha256'] == audit['replay_sha256'] == matched['cases'][arm]['replay_sha256'], arm
        assert digest(folder / 'navigation.mp4') == media[arm]['video_sha256'] == render['video_sha256'], arm
        assert digest(folder / 'track.gpx') == render['gpx_sha256'], arm
        assert digest(STUDY / 'reproduce/render_round_trip.py') == render['render_script_sha256'], arm
        assert digest(folder / 'departure_gate.jsonl') == gates[arm]['gate_log_sha256'] == audit['departure_gate_log_sha256'] == exposures[arm]['gate_log_sha256'], arm
        assert digest(folder / 'departure_gate.json') == audit['departure_gate_summary_sha256'], arm
        assert execution['return_code'] == 0 and execution['source_unchanged'] and not execution['changed_inputs'], arm
        assert render['launch_summary_frames'] > 0
        assert render['actual_departure_utc'] == gate['actual_departure_utc'] == gates[arm]['actual_departure_utc']
        assert render['gate_wait_seconds'] == gate['gate_wait_seconds']
        assert render['retest_config_sha256'] == digest(STUDY / 'underway_config.json')
        assert render['underway_fixture_sha256'] == execution['underway_traffic_fixture_sha256'] == audit['underway_truth_fixture_sha256']
        assert render['underway_truth_trace_sha256'] == execution['underway_truth_trace_sha256'] == audit['underway_truth_trace_sha256']
        for key in ('underway_first_emission_s', 'underway_first_delivery_s', 'underway_reference_elapsed_s',
                    'underway_drop_emission_windows_s', 'guidance_objective', 'guidance_label'):
            assert render[key] == execution[key], (arm, key)
        assert render['underway_drop_emission_windows_s'] == config['underway_drop_emission_windows_s']
        assert render['underway_retention_original_epoch_seconds'] == config['underway_frame_retention_original_epoch_s']
        count = audit['applied_actions'] + 1
        assert render['source_frame_count'] == media[arm]['source_frames'] == count
        start, stop = config['underway_frame_retention_original_epoch_s']
        # The independent native audit verifies the five-second frame clock.
        launch_encounter_s = gate['gate_wait_seconds'] - 40
        expected = [step for step in range(count) if start <= launch_encounter_s + step * 5 <= stop]
        assert render['underway_retained_source_steps'] == expected, arm
        assert set(expected).issubset(render['source_frame_indices']), arm
        assert media[arm]['tactical_updates_retained'] == len(expected), arm
        assert render['underway_full_window_recorded'] == (launch_encounter_s + (count - 1) * 5 >= stop)
        assert exposures[arm]['both_actual65s_underway_peaks_verified'] == audit['underway_both65s_age_peaks_verified']
        assert exposures[arm]['underway_contact_first_acquisition_s'] == audit['underway_first_acquisition_encounter_elapsed_s']
        assert exposures[arm]['active_AIS_or_COLREG_steps_420_to900'] == audit['underway_active_ais_or_colreg_steps_420_to_900_s']
        tactical_updates += len(expected)
    for destination in re.findall(r'\]\(([^:()]+)\)', (STUDY / 'playback.md').read_text()):
        assert (STUDY / destination).is_file(), destination
    paper = read('paper_layout_verification.json')
    package = read('paper_package_verification.json')
    assert digest(PAPER / 'manuscript.pdf') == paper['pdf_sha256']
    assert digest(PAPER / 'journal_manuscript_sources.zip') == paper['source_zip_sha256'] == package['source_zip_sha256']
    assert paper['all_pages_rendered_and_visually_reviewed'] and package['exact_source_byte_parity'] and package['zip_integrity']
    with zipfile.ZipFile(PAPER / 'journal_manuscript_sources.zip') as archive:
        assert archive.testzip() is None
        names = archive.namelist()
        assert len(names) == len(set(names)) == paper['source_zip_files'] == package['source_zip_files']
        for name in names:
            assert not Path(name).is_absolute() and '..' not in Path(name).parts
            assert archive.read(name) == (PAPER / name).read_bytes(), name
            assert digest(PAPER / name) == package['source_files'][name], name
    manifest = read('audit_outputs_sha256.json')
    assert manifest and all(digest(STUDY / name) == checksum for name, checksum in manifest.items())
    rows = list(audits.values())
    summary = {
        'study_and_artifact_delivery_complete': True,
        'combined_acceptance_passed': acceptance['all_passed'],
        'verified_cases': len(audits),
        'completed_ordered_50m_loops': completed,
        'verified_50m_captures': sum(len(r['captures']) for r in comparison.values()),
        'swept_applied_commands': sum(r['applied_actions'] for r in rows),
        'selection_attempts_through_accept_or_stop': sum(r['AIS_applied_decision_attempts'] for r in rows),
        'gate_attempts': sum(r['gate_attempts'] for r in comparison.values()),
        'maximum_gate_s': max(r['maximum_gate_decision_s'] for r in comparison.values()),
        'maximum_complete_selection_s': extrema(rows, 'maximum_complete_AIS_applied_decision_s', maximum=True),
        'maximum_recorded_regatta_search_s': extrema(rows, 'maximum_recorded_regatta_search_wall_s', maximum=True),
        'minimum_truth_full_plus_reserve_m': extrema(rows, 'configured_truth_centered_minimum_m'),
        'minimum_report_full_plus_reserve_m': extrema(rows, 'configured_model_minimum_m'),
        'fixed_traffic_faults_and_settings_verified': True,
        'underway_contact_acquired_at420s_cases': sum(r['underway_first_acquisition_encounter_elapsed_s'] == 420 for r in rows),
        'both_underway65s_peaks_verified_cases': sum(r['underway_both65s_age_peaks_verified'] for r in rows),
        'both_underway40s_losses_experienced_cases': sum(r['both_full40s_underway_lost_delivery_effects_experienced'] for r in exposures.values()),
        'active_underway_AIS_or_COLREG_cases': sum(r['underway_active_ais_or_colreg_steps_420_to_900_s'] > 0 for r in rows),
        'retained_underway_tactical_updates': tactical_updates,
        'sheltered_staging_scope': 'Protected external staging; no simulated berth station keeping. The new encounter and loss exposure are checked underway. Before acquisition, new-contact truth allowance is labeled hypothetical; no report projection is fabricated.',
        'verified_video_replay_gate_sets': len(media),
        'manifest_files_hashed_and_verified': len(manifest),
        'paper': paper,
        'receipts': {name: digest(STUDY / name) for name in RECEIPTS},
        'verification_script_sha256': digest(Path(__file__)),
    }
    assert summary_receipt['applied_commands'] == summary['swept_applied_commands']
    assert summary_receipt['verified_captures'] == summary['verified_50m_captures']
    assert summary_receipt['completed_loops'] == completed
    (STUDY / 'deliverable_verification.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: v for k, v in summary.items() if k not in ('paper', 'receipts')}, indent=2))


if __name__ == '__main__':
    main()

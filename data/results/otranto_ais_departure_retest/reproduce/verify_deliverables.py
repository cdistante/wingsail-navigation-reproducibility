"""Check outputs and bind matched traffic, navigation, media and paper evidence."""
from pathlib import Path
import hashlib
import json
import re
import zipfile

STUDY = Path(__file__).resolve().parents[1]
PAPER = STUDY.parents[2] / 'reports/journal_manuscript'
EXPECTED_ORDER = [
    ('outbound', '002'), ('outbound', '003'), ('outbound', '004'), ('outbound', '005'),
    ('return', '004'), ('return', '003'), ('return', '002'), ('return', '001'),
]


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(name):
    return json.loads((STUDY / name).read_text())


def main():
    comparison = read('comparison.json')['runs']
    acceptance = read('independent_acceptance_check.json')
    audits = {row['case']: row for row in read('independent_traffic_sweep.json')}
    gates = {row['case']: row for row in read('independent_departure_gate_check.json')['cases']}
    media = read('media_verification.json')['runs']
    assert len(comparison) == 8 and set(comparison) == set(audits) == set(gates) == set(media)
    assert read('matched_inputs_verification.json')['all_cases_match_original_inputs']
    assert read('baseline_preservation_check.json')['all_preserved']
    assert read('fault_exposure_check.json')['all_cases_final']
    ordered = 0
    for arm, run in comparison.items():
        captures = run['captures']
        actual_order = [(c['phase'], c['waypoint']) for c in captures]
        assert actual_order == EXPECTED_ORDER[:len(actual_order)]
        assert all(c['distance_metres'] <= 50 for c in captures)
        ordered += actual_order == EXPECTED_ORDER and run['status'] == 'completed'
        folder = STUDY / 'runs' / arm
        assert digest(folder / 'navigation.jsonl.gz') == media[arm]['replay_sha256'] == audits[arm]['replay_sha256']
        assert digest(folder / 'navigation.mp4') == media[arm]['video_sha256']
        render = json.loads((folder / 'video_verification.json').read_text())
        assert render['launch_summary_frames'] > 0
        assert digest(folder / 'departure_gate.jsonl') == gates[arm]['gate_log_sha256']
    for destination in re.findall(r'\]\(([^:()]+)\)', (STUDY / 'playback.md').read_text()):
        assert (STUDY / destination).is_file(), destination
    paper = read('paper_layout_verification.json')
    assert digest(PAPER / 'manuscript.pdf') == paper['pdf_sha256']
    assert digest(PAPER / 'journal_manuscript_sources.zip') == paper['source_zip_sha256']
    with zipfile.ZipFile(PAPER / 'journal_manuscript_sources.zip') as archive:
        assert archive.testzip() is None
        assert all(archive.read(name) == (PAPER / name).read_bytes() for name in archive.namelist())
    manifest = read('audit_outputs_sha256.json')
    assert all(digest(STUDY / name) == value for name, value in manifest.items())
    summary = {
        'study_and_artifact_delivery_complete': True,
        'combined_acceptance_passed': acceptance['all_passed'],
        'verified_cases': len(audits),
        'completed_ordered_50m_loops': ordered,
        'verified_50m_captures': sum(len(r['captures']) for r in comparison.values()),
        'swept_applied_commands': sum(r['applied_actions'] for r in audits.values()),
        'selection_attempts_through_accept_or_stop': sum(r['AIS_applied_decision_attempts'] for r in audits.values()),
        'gate_attempts': sum(r['gate_attempts'] for r in comparison.values()),
        'maximum_gate_s': max(r['maximum_gate_decision_s'] for r in comparison.values()),
        'maximum_complete_selection_s': max(r['maximum_complete_AIS_applied_decision_s'] for r in audits.values()),
        'minimum_truth_full_plus_reserve_m': min(r['configured_truth_centered_minimum_m'] for r in audits.values()),
        'minimum_report_full_plus_reserve_m': min(r['configured_model_minimum_m'] for r in audits.values()),
        'fixed_traffic_faults_and_settings_verified': True,
        'sheltered_staging_scope': 'No simulated open-water berth or physical station keeping. Fault exposure and clearance apply to separate phases.',
        'verified_video_replay_gate_sets': len(media),
        'manifest_files_hashed_and_verified': len(manifest),
        'paper': paper,
        'receipts': {name: digest(STUDY / name) for name in (
            'comparison.json', 'independent_acceptance_check.json', 'independent_departure_gate_check.json',
            'independent_traffic_sweep.json', 'fault_exposure_check.json', 'matched_inputs_verification.json',
            'audit_outputs_sha256.json', 'video_render.log', 'media_verification.json', 'paper_layout_verification.json',
        )},
    }
    (STUDY / 'deliverable_verification.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({k:v for k,v in summary.items() if k not in ('paper', 'receipts')}, indent=2))


if __name__ == '__main__':
    main()

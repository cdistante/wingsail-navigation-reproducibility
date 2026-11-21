"""Bind final navigation, acceptance, paper and media checks for the retest."""
from pathlib import Path
import hashlib
import json
import re
import zipfile

STUDY = Path(__file__).resolve().parents[1]
PAPER = STUDY.parents[2] / 'reports/journal_manuscript'


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(name):
    return json.loads((STUDY / name).read_text())


def main():
    audits = read('independent_traffic_sweep.json')
    acceptance = read('independent_acceptance_check.json')
    matched = read('matched_inputs_verification.json')
    media = read('media_verification.json')['runs']
    assert len(audits) == len(acceptance['cases']) == len(media) == 8
    assert matched['all_cases_match_original_inputs'] and read('baseline_preservation_check.json')['all_preserved']
    assert not acceptance['all_passed']
    cleared = sum(a['configured_truth_centered_minimum_m'] > 0 and a['configured_model_minimum_m'] > 0 for a in audits)
    met_timing = sum(a['all_AIS_choose_attempts_below_5s'] and a['all_AIS_applied_decision_attempts_below_5s'] for a in audits)
    completed = sum(a['mission_status'] == 'completed' for a in audits)
    full_peak_count = sum(bool(a['longest_fault_peak_experienced']) for a in audits)
    assert (cleared, met_timing, completed, full_peak_count) == (8, 8, 0, 0)
    assert all(not a['goal_captures'] and a['mission_status'] == 'no_full_clearance_action' for a in audits)
    applied = sum(a['applied_actions'] for a in audits)
    attempts = sum(a['AIS_applied_decision_attempts'] for a in audits)
    assert applied == 226 and attempts == 234
    output_proof = read('paper_layout_verification.json')
    assert output_proof['pdf_sha256'] == digest(PAPER / 'manuscript.pdf')
    assert output_proof['source_zip_sha256'] == digest(PAPER / 'journal_manuscript_sources.zip')
    with zipfile.ZipFile(PAPER / 'journal_manuscript_sources.zip') as archive:
        assert archive.testzip() is None and all(archive.read(name) == (PAPER / name).read_bytes() for name in archive.namelist())
    assert all(digest(STUDY / 'runs' / arm / 'navigation.mp4') == proof['video_sha256'] for arm, proof in media.items())
    assert all(digest(STUDY / 'runs' / arm / 'navigation.jsonl.gz') == proof['replay_sha256'] for arm, proof in media.items())
    assert all(proof['source_frames'] == proof['unique_physical_steps'] for proof in media.values())
    for destination in re.findall(r'\]\(([^:()]+)\)', (STUDY / 'playback.md').read_text()):
        assert (STUDY / destination).is_file(), destination
    manifest = read('audit_outputs_sha256.json')
    assert all(digest(STUDY / name) == value for name, value in manifest.items())
    result = {
        'study_execution_and_artifact_delivery_complete': True,
        'controller_combined_acceptance_passed': False,
        'retained_cases': 8, 'strict_full_clearance_recorded_prefixes': cleared,
        'AIS_selection_under_five_seconds': met_timing, 'ordered_full_routes': completed,
        'exposed_maximum_65s_report_age': full_peak_count,
        'applied_commands': applied, 'decision_attempts_including_terminal': attempts,
        'independently_reconstructed_native_substeps': 100 * applied,
        'maximum_AIS_selection_through_accept_or_stop_s': max(a['maximum_complete_AIS_applied_decision_s'] for a in audits),
        'smallest_truth_full_plus_reserve_margin_m': min(a['configured_truth_centered_minimum_m'] for a in audits),
        'smallest_report_full_plus_reserve_margin_m': min(a['configured_model_minimum_m'] for a in audits),
        'replay_and_decoded_video_pairs': len(media), 'paper': output_proof,
        'verified_study_numeric_and_source_files': len(manifest),
        'receipts': {name: digest(STUDY / name) for name in (
            'independent_traffic_sweep.json', 'independent_acceptance_check.json', 'matched_inputs_verification.json',
            'baseline_preservation_check.json', 'audit_outputs_sha256.json', 'media_verification.json',
            'comparison.json', 'baseline_comparison.csv', 'telemetry.csv', 'paper_layout_verification.json')},
    }
    (STUDY / 'deliverable_verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print('Verified eight matched replays, decoded MP4s, independent outcomes and manuscript exports.')


if __name__ == '__main__':
    main()

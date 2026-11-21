"""Independently verify received-AIS departure decisions and their fixed clock.

No actor gate, AISPrediction or runtime traffic-provider scorer is called.
Sheltered monitoring is not treated as an open-water vessel trajectory.
"""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import numpy as np

from audit_independent import (DT, START, TRAFFIC_EPOCH, Truth, digest,
                               expected_report_age, local_east_north_metres,
                               swept_margin, velocity, save_json)

STUDY = Path(__file__).resolve().parents[1]
EXPECTED_SETTINGS = {
    'maximum_report_age_s': 20.0,
    'uncapped_startup_seconds': 120.0,
    'preview_seconds': 300.0,
    'preview_age_budget_s': 65.0,
    'startup_speed_bound_mps': 2.0,
    'startup_room_m': 30.0,
    'required_consecutive_observations': 3,
    'maximum_wait_s': 1800.0,
    'interval_s': 5.0,
    'acceleration_bound_mps2': .026,
    'clearance_reserve_metres': 3.0,
}


def timestamp(value):
    return np.datetime64(value.removesuffix('Z'), 'ns')


def tube_margin(centres, velocities, ages, hulls, horizon, parameters, capped):
    times = np.r_[np.arange(0., horizon, DT), horizon]
    caps = np.maximum(DT, parameters['preview_age_budget_s'] - ages)
    if capped:
        times = np.unique(np.r_[times, caps[(caps > 0) & (caps < horizon)]])
    elapsed = times[:, None]
    if capped:
        elapsed = np.minimum(elapsed, caps)
    age = ages + elapsed
    uncertainty = 5. + np.maximum(.05 * age, .5 * .026 * age**2)
    radii = (hulls + 25. + uncertainty + parameters['clearance_reserve_metres']
             + parameters['startup_room_m'] + parameters['startup_speed_bound_mps'] * times[:, None])
    projected = centres + times[:, None, None] * velocities
    # Only chord fractions affect the minimum; nonuniform cap knots do not
    # require the elapsed-time output of this independent analytic minimizer.
    margins, _ = swept_margin(projected, radii, 1.)
    return float(margins.min())


def audit_case(run, monitoring_only=False):
    metadata = json.loads((run / 'experiment.json').read_text())
    summary_path, log_path = run / 'departure_gate.json', run / 'departure_gate.jsonl'
    summary = json.loads(summary_path.read_text())
    rows = [json.loads(line) for line in log_path.read_text().splitlines()]
    parameters = summary['parameters']
    assert parameters == EXPECTED_SETTINGS, (run.name, 'Gate parameters differ from frozen specification')
    assert timestamp(summary['monitoring_start_utc']) == START
    assert timestamp(summary['traffic_epoch_utc']) == TRAFFIC_EPOCH
    assert summary['log_sha256'] == digest(log_path)
    assert summary['attempts'] == len(rows) and rows
    fixture, trace = Path(metadata['traffic_fixture']), Path(metadata['truth_trace'])
    truth = Truth(fixture, trace)
    identities = truth.identities
    initial_state = rows[0]['own_state']
    np.testing.assert_allclose([initial_state['latitude'], initial_state['longitude']], truth.origin, atol=1e-9, rtol=0)
    assert initial_state['heading'] == 45. and initial_state['speed'] == 0. and initial_state['flap'] == -10.
    previous_generations, roster = {}, set()
    streak = 0
    first_launch = None
    maximum_formula_error = maximum_report_position_error = maximum_report_velocity_error = 0.
    all_margins = []
    max_observed_age = max_held_age = 0.
    timing = []
    for step, row in enumerate(rows):
        assert row['step'] == step and row['monitoring_elapsed_s'] == step * DT
        assert row['own_state'] == initial_state, (run.name, step, 'Vessel moved during sheltered staging')
        when = START + np.timedelta64(round(step * DT * 1e9), 'ns')
        assert timestamp(row['observation_utc']) == when
        elapsed = float((when - TRAFFIC_EPOCH) / np.timedelta64(1, 's'))
        reports_by_id = {report['mmsi']: report for report in row['received_reports']}
        assert len(reports_by_id) == len(row['received_reports'])
        assert set(reports_by_id) == set(identities), (run.name, step, 'Incomplete fixed fleet roster')
        reports = [reports_by_id[mmsi] for mmsi in identities]
        ages = np.asarray([report['report_age_seconds'] for report in reports])
        np.testing.assert_allclose(ages, expected_report_age(elapsed), atol=1e-8, rtol=0)
        raw_positions = np.column_stack(local_east_north_metres(
            [r['latitude'] for r in reports], [r['longitude'] for r in reports], *truth.origin))
        velocities = velocity([r['course'] for r in reports], [r['speed'] for r in reports])
        for column, report in enumerate(reports):
            generation = elapsed - ages[column]
            true_position = truth.sample([generation], (1, 2))[0, column]
            true_velocity = truth.sample([generation], (3, 4))[0, column]
            position_error = float(np.linalg.norm(raw_positions[column] - true_position))
            velocity_error = float(np.linalg.norm(velocities[column] - true_velocity))
            maximum_report_position_error = max(maximum_report_position_error, position_error)
            maximum_report_velocity_error = max(maximum_report_velocity_error, velocity_error)
            assert position_error < .001 and velocity_error < 1e-8
        centres = raw_positions + ages[:, None] * velocities
        uncertainty = 5. + np.maximum(.05 * ages, .5 * .026 * ages**2)
        distance = np.linalg.norm(centres, axis=1)
        speed = np.linalg.norm(velocities, axis=1)
        robust = (np.sum(centres * velocities, axis=1) - 2. * speed * uncertainty) / np.maximum(distance, np.finfo(float).tiny) - .026 * ages
        robust = np.where(distance > uncertainty, robust, -1.)
        calculated = {
            'min_robust_receding_mps': float(robust.min()),
            'min_uncapped_startup_margin_m': tube_margin(centres, velocities, ages, truth.hulls, 120., parameters, False),
            'min_preview_margin_m': tube_margin(centres, velocities, ages, truth.hulls, 300., parameters, True),
        }
        for key, value in calculated.items():
            difference = abs(value - row[key])
            maximum_formula_error = max(maximum_formula_error, difference)
            assert difference < 1e-6, (run.name, step, key, value, row[key])
        fresh = bool(np.all(ages <= parameters['maximum_report_age_s']))
        roster.update(reports_by_id)
        continuous = bool(roster) and roster == set(reports_by_id)
        generations = {mmsi: step * DT - reports_by_id[mmsi]['report_age_seconds'] for mmsi in identities}
        renewed = all(value > previous_generations.get(mmsi, -np.inf) for mmsi, value in generations.items())
        previous_generations.update(generations)
        eligible = fresh and all(value > 0. for value in calculated.values())
        streak = streak + 1 if eligible and continuous and renewed else 0
        launch = streak >= parameters['required_consecutive_observations']
        assert row['fresh_reports'] == fresh and row['roster_continuous'] == continuous
        assert row['reports_renewed'] == renewed and row['clear_streak'] == streak
        assert row['launched'] == launch
        if launch and first_launch is None:
            first_launch = step
        assert not launch or step == len(rows) - 1, 'Monitoring continued beyond first qualifying departure'
        wall = row['wall_seconds']
        assert np.isfinite(wall) and wall >= 0.
        timing.append(wall)
        max_observed_age = max(max_observed_age, float(ages.max()))
        if step < len(rows) - 1:
            max_held_age = max(max_held_age, float(ages.max()) + DT)
        all_margins.append(calculated)
    selected = first_launch is not None
    assert summary['gate_wait_seconds'] == (len(rows) - 1) * DT
    assert abs(summary['maximum_gate_decision_s'] - max(timing)) < 1e-12
    assert (summary['actual_departure_utc'] is not None) == selected
    initial_matches = None
    if selected:
        assert timestamp(summary['actual_departure_utc']) == START + np.timedelta64(round(summary['gate_wait_seconds'] * 1e9), 'ns')
        assert summary['certificate'] == rows[-1]
        if not monitoring_only:
            with gzip.open(run / 'navigation.jsonl.gz', 'rt') as stream:
                next(stream)
                initial = json.loads(next(stream))
            assert timestamp(initial['observation']['time']) == timestamp(summary['actual_departure_utc'])
            assert initial['state'] == initial_state == initial['observation']['boat']
            received = {r['mmsi']: r for r in initial['observation']['ais_targets']}
            assert received == {r['mmsi']: r for r in rows[-1]['received_reports']}
            initial_matches = True
    else:
        assert summary['certificate'] is None and summary['gate_wait_seconds'] == parameters['maximum_wait_s']
    return {
        'case': run.name, 'encounter_id': metadata['encounter_id'],
        'scope': 'received-report sheltered monitoring; no prelaunch plant motion or in-water clearance is claimed',
        'launch_selected': selected, 'first_qualifying_gate_step': first_launch,
        'monitoring_start_utc': summary['monitoring_start_utc'], 'traffic_epoch_utc': summary['traffic_epoch_utc'],
        'actual_departure_utc': summary['actual_departure_utc'], 'gate_wait_seconds': summary['gate_wait_seconds'],
        'gate_attempts': len(rows), 'maximum_gate_decision_s': max(timing),
        'all_gate_attempts_below_5s': all(value < 5. for value in timing),
        'maximum_instantaneous_monitored_report_age_s': max_observed_age,
        'maximum_monitored_held_age_to_next_poll_s': max_held_age,
        '65s_report_age_peak_experienced_only_while_sheltered': max_held_age >= 65. - 1e-8,
        'maximum_independent_formula_difference': maximum_formula_error,
        'maximum_independent_report_position_error_m': maximum_report_position_error,
        'maximum_independent_report_velocity_error_mps': maximum_report_velocity_error,
        'final_launch_certificate': all_margins[-1] if selected else None,
        'fixed_traffic_and_reporting_clock_verified': True,
        'first_three_consecutive_fresh_eligible_reports_verified': True,
        'sheltered_own_state_unchanged': True, 'selected_state_and_reports_equal_launch_replay': initial_matches,
        'gate_log_sha256': digest(log_path), 'gate_summary_sha256': digest(summary_path),
        'fixture_sha256': digest(fixture), 'truth_trace_sha256': digest(trace),
        'audit_source_sha256': digest(__file__),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, default=STUDY)
    parser.add_argument('--case', action='append', default=[])
    parser.add_argument('--allow-live', action='store_true', help='Include active runs whose launch gate and initial replay frame are available.')
    parser.add_argument('--monitoring-only', action='store_true', help='Audit completed gate logs before the native replay is closed; skip selected-row/replay binding.')
    args = parser.parse_args()
    study = args.study.resolve()
    runs = sorted((study / 'runs').glob('*')) if args.monitoring_only or args.allow_live else sorted(path.parent for path in (study / 'runs').glob('*/execution.json'))
    if args.case:
        runs = [run for run in runs if run.name in args.case]
        assert {run.name for run in runs} == set(args.case)
    assert runs, 'No selected departure receipts are available.'
    records = [audit_case(run, args.monitoring_only) for run in runs]
    output = study / ('independent_departure_gate_monitoring_check.json' if args.monitoring_only else 'independent_departure_gate_check.json')
    save_json(output, {'all_available_gates_verified': True, 'monitoring_only': args.monitoring_only,
                      'all_gate_attempts_below_5s': all(r['all_gate_attempts_below_5s'] for r in records),
                      'source_sha256': digest(__file__), 'cases': records})
    for row in records:
        print(row['case'], row['gate_wait_seconds'], row['final_launch_certificate'], flush=True)


if __name__ == '__main__':
    main()

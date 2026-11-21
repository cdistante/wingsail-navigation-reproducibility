"""Build immutable one-vessel follow-on encounters from a documented design."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

STUDY = Path(__file__).resolve().parents[1]
FIRST_EMISSION_SECONDS = 400.0


def velocity(course_degrees, speed):
    angle = np.radians(course_degrees)
    return speed * np.array([np.sin(angle), np.cos(angle)])


def build(scenario, plan):
    directory = STUDY / 'underway_fixtures' / scenario
    directory.mkdir(parents=True, exist_ok=True)
    origin = json.loads((STUDY / 'fixtures' / scenario / 'ground_truth.json').read_text())['origin_latitude_longitude']
    anchor = np.asarray(plan['unmanoeuvred_anchor_en_m'])
    v0 = velocity(plan['course_deg'], plan['speed_mps'])
    v1 = velocity(plan['final_course_deg'], plan['final_speed_mps'])
    p0 = anchor - plan['unmanoeuvred_anchor_s'] * v0
    t0, t1 = plan['ramp_start_s'], plan['ramp_end_s']
    acceleration = (v1 - v0) / (t1 - t0)
    rate = float(np.linalg.norm(acceleration))
    assert 0 < rate < .026
    vessel = {
        'mmsi': plan['mmsi'], 'name': f'VALIDATION {scenario.upper()} UNDERWAY CONTACT',
        'callsign': f"UW{plan['mmsi'] % 1000:d}", 'ship_type': 70,
        'initial_east_metres': float(p0[0]), 'initial_north_metres': float(p0[1]),
        'speed_metres_per_second': plan['speed_mps'], 'course_degrees': plan['course_deg'],
        'length_metres': plan['length_m'], 'beam_metres': plan['beam_m'],
        'destination': 'OFFLINE VALIDATION', 'heading_degrees': None,
    }
    fixture = {
        'fixture': f'{scenario}_underway', 'source': 'Predefined follow-on encounter using earlier traffic-free prefix kinematics for design only.',
        'position_model': 'Piecewise constant velocity with one linear velocity ramp in east/north.',
        'first_emission_s': FIRST_EMISSION_SECONDS,
        'origin_latitude_longitude': origin, 'scenarios': {'mixed': [vessel]},
        'velocity_ramps': [{
            'mmsi': plan['mmsi'], 'start_s': t0, 'end_s': t1,
            'final_course_degrees': plan['final_course_deg'], 'final_speed_mps': plan['final_speed_mps'],
            'acceleration_magnitude_mps2': rate,
        }],
        'design_anchors': [{
            'mmsi': plan['mmsi'], 'unmanoeuvred_anchor_s': plan['unmanoeuvred_anchor_s'],
            'east_m': float(anchor[0]), 'north_m': float(anchor[1]),
        }],
    }
    (directory / 'ground_truth.json').write_text(json.dumps(fixture, indent=2, allow_nan=False) + '\n')
    times = np.arange(-1600, 24001) / 20
    turning = np.clip(times - t0, 0., t1 - t0)
    after = np.maximum(times - t1, 0.)
    positions = p0 + times[:, None] * v0 + (0.5*turning**2 + (t1 - t0)*after)[:, None] * acceleration
    motions = v0 + turning[:, None] * acceleration
    with (directory / 'truth_trace.csv').open('w', newline='') as stream:
        writer = csv.writer(stream, lineterminator='\n')
        writer.writerow(('elapsed_s','mmsi','east_m','north_m','east_velocity_mps','north_velocity_mps'))
        writer.writerows((f'{t:.2f}', plan['mmsi'], *(f'{x:.12f}' for x in (*p,*v)))
                         for t,p,v in zip(times,positions,motions,strict=True))
    print(scenario, 'initial EN', tuple(p0), 'max acceleration', rate)


def main():
    specification = json.loads((STUDY / 'underway_fixture_design.json').read_text())
    for scenario, plan in specification['encounters'].items():
        build(scenario, plan)


if __name__ == '__main__':
    main()

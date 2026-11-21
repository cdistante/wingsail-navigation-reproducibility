"""Audit launch ground VMG for exact beam reaches and the shared steady polar."""
from pathlib import Path
import csv
import gzip
import json
import math

import numpy as np

from world._geo import local_east_north_metres
from world.drift_guidance import DriftGuidance, DriftMode, gate_velocity
from world.forecast_tacks import steady_sailing_trim
from world.fossen import Fossen3DOFSaildroneDynamics
from world.navigation_replay import _decode_episode_frame

STUDY = Path(__file__).resolve().parents[1]


def evaluate(arm, dynamics, guide):
    with gzip.open(STUDY / 'runs' / arm / 'navigation.jsonl.gz', 'rt') as stream:
        header = json.loads(next(stream))
        first = _decode_episode_frame(json.loads(next(stream)))
    observation = first.observation
    mark = next(p for p in header['route'] if p['name'] == '002')
    displacement = np.array(local_east_north_metres(
        mark['latitude'], mark['longitude'], first.state.latitude, first.state.longitude,
    ))
    axis = displacement / np.linalg.norm(displacement)
    current = np.array([observation.current.u, observation.current.v])
    steady = tuple(guide._steady_modes(observation))
    modes = [(mode, float(mode.velocity @ axis)) for mode in steady]
    gate = gate_velocity(np.array([mode.velocity for mode in steady]), displacement)
    p = dynamics.parameters
    for offset in (-90.0, 90.0):
        bow = (observation.wind.direction + offset) % 360.0
        rotation = dynamics._body_to_earth(bow)
        port_starboard = []
        for flap in np.linspace(p.flap_min, p.flap_max, 7):
            trim = steady_sailing_trim(observation, dynamics, bow, flap)
            if trim is not None:
                water, rudder = trim
                port_starboard.append(DriftMode(rotation @ water + current, bow, rudder, float(flap)))
        mode = max(port_starboard, key=lambda m: m.velocity @ axis, default=None)
        closest_seconds = None
        closest_range = None
        if mode is not None:
            speed_squared = float(mode.velocity @ mode.velocity)
            closest_seconds = max(0.0, float(displacement @ mode.velocity) / speed_squared) if speed_squared else 0.0
            closest_range = float(np.linalg.norm(displacement - closest_seconds * mode.velocity))
        row = {'arm': arm, 'utc': str(observation.time) + 'Z',
               'wind_from_deg': observation.wind.direction, 'wind_mps': observation.wind.speed,
               'current_to_deg': observation.current.direction, 'current_mps': observation.current.speed,
               'mark_bearing_deg': math.degrees(math.atan2(*displacement)) % 360.0,
               'current_vmg_mps': float(current @ axis), 'beam_heading_deg': bow,
               'beam_offset_deg': offset, 'beam_feasible_count': len(port_starboard),
               'beam_best_flap_deg': mode.flap if mode else None,
               'beam_best_cog_deg': mode.course if mode else None,
               'beam_best_speed_water_mps': float(np.linalg.norm(mode.velocity-current)) if mode else None,
               'beam_best_speed_ground_mps': float(np.linalg.norm(mode.velocity)) if mode else None,
               'beam_best_vmg_mps': float(mode.velocity @ axis) if mode else None,
               'beam_constant_velocity_closest_range_m': closest_range,
               'beam_constant_velocity_time_of_closest_s': closest_seconds,
               'best_sampled_instantaneous_vmg_mps': max(vmg for _, vmg in modes),
               'best_no_cross_track_net_vmg_mps': gate}
        yield row


def main():
    dynamics = Fossen3DOFSaildroneDynamics()
    guide = DriftGuidance(dynamics, 45.0)
    rows = [row for folder in sorted((STUDY / 'runs').glob('*__classical'))
            for row in evaluate(folder.name, dynamics, guide)]
    assert len(rows) == 8
    path = STUDY / 'beam_reach_at_launch.csv'
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(row['arm'], row['beam_heading_deg'],
              'ground VMG', row['beam_best_vmg_mps'], 'balanced VMG', row['best_no_cross_track_net_vmg_mps'])


if __name__ == '__main__':
    main()

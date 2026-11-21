"""Register and screen matched native-counterfactual starts without running labels."""

from __future__ import annotations

from copy import copy, deepcopy
from functools import lru_cache
from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np


STUDY = Path(__file__).resolve().parents[1]
CAUSAL = STUDY.with_name("temporal_embedding_native")
sys.path.insert(0, str(CAUSAL / "reproduce"))

from run_temporal_ablation import (
    MANIFEST, TRUST_REGRET_S, AsOfSeasonalArchive, CausalEfficientNativeEngine, CausalScenarioEnvironment,
    forecasts, gate_indices,
)
from native_branches import NativeConfig, NativeSnapshot
from scenario_environment import CHART_PATH, ScenarioEnvironment, _chart_grid, _true_bearing
from seasonal_environment import SeasonalArchive
from world._geo import local_east_north_metres, normalise_time, offset_latitude_longitude
from world.boat import BoatState
from world.scenario import RouteWaypoint


SPLITS = ("train", "calibration")
WEEKS = {"train": ("winter", "summer"), "calibration": ("spring",)}
SOURCE_ROUTE = "otranto_003"
CAP_S = 4_800
CURRENT_CUTOFF_MPS = -.15
MIN_ALTERNATIVES = 2
NATIVE_CONFIG = NativeConfig()
# Only 001→002 and 002→003 are source legs; no Brindisi, October,
# 004→005, or participant-selected cases enter either development split.
VARIANTS = (
    ((0, 1.), (-15, 1.), (15, 1.)),
    ((-30, 1.), (30, 1.), (-15, .85), (15, 1.15)),
    ((0, .85), (0, 1.15), (-30, 1.15), (30, .85)),
)
MAX_EXPANSION = 3
TRAFFIC_NAMES = {
    "train": ("train_crossing", "train_opposed"),
    "calibration": ("validation_slanted",),
}
INITIAL_TRIMS = (
    (0., 0., -10.), (-30., .28, -10.), (30., .28, 10.),
    (-55., .45, -10.), (55., .45, 10.),
)


@lru_cache(maxsize=1)
def manifest():
    return json.loads(MANIFEST.read_text())


@lru_cache(maxsize=3)
def archives(block):
    """Retain interpolated forcing for the plant and as-of slices for decisions."""
    parent = MANIFEST.parent
    interpolated = SeasonalArchive(parent / "forcing" / block)
    return interpolated, AsOfSeasonalArchive(interpolated)


def coordinates(point):
    return point["latitude"], point["longitude"]


def base_point(point):
    return {key: point[key] for key in ("waypoint", "latitude", "longitude")}


def wet_chart(points):
    chart, axes = _chart_grid(CHART_PATH)
    indices = np.rint((np.asarray(points) - axes[:, 0]) / axes[:, 1])
    if not np.all((indices >= 0) & (indices < np.asarray(chart.shape))):
        return False
    rows, columns = indices.astype(np.intp).T
    return bool(np.isfinite(chart.elevation[rows, columns]).all() and not chart.occupancy[rows, columns].any())


def rotated_goal(origin, goal, turn_degrees, distance_factor):
    if turn_degrees == 0 and distance_factor == 1.:
        return goal.copy()
    east, north = local_east_north_metres(*coordinates(goal), *coordinates(origin))
    angle = np.deg2rad(turn_degrees)
    east, north = np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]]) @ (east, north)
    latitude, longitude = offset_latitude_longitude(*coordinates(origin), east * distance_factor, north * distance_factor)
    return {"waypoint": goal["waypoint"], "latitude": float(latitude), "longitude": float(longitude)}


def geometry_variants(expansion):
    records = []
    route = manifest()["routes"][SOURCE_ROUTE]
    waypoints = tuple(map(base_point, route["waypoints"]))
    for leg, (origin, original_goal) in enumerate(zip(waypoints, waypoints[1:]), start=1):
        for turn, factor in (variant for tier in VARIANTS[:expansion + 1] for variant in tier):
            goal = rotated_goal(origin, original_goal, turn, factor)
            bearing = _true_bearing(coordinates(origin), coordinates(goal))
            distance = float(np.hypot(*local_east_north_metres(*coordinates(goal), *coordinates(origin))))
            radius = NATIVE_CONFIG.capture_radius_m
            azimuth = np.linspace(0., 2. * np.pi, 33)
            lat, lon = offset_latitude_longitude(*coordinates(goal), radius * np.sin(azimuth), radius * np.cos(azimuth))
            corridor = np.vstack((np.linspace(coordinates(origin), coordinates(goal), 41),
                                  np.column_stack((lat, lon))))
            if wet_chart(corridor):
                records.append({
                    "route_id": f"otranto_leg{leg}_turn{turn:+03}_scale{int(round(factor * 100)):03}",
                    "leg": f'{origin["waypoint"]}-{goal["waypoint"]}',
                    "route_family": route["family"], "route_gpx": route["route_gpx"],
                    "departure": origin.copy(), "goal": goal,
                    "bearing_deg": bearing, "distance_m": distance,
                    "rotation_deg": turn, "distance_scale": factor,
                })
    return records


def conditions(observation, bearing_deg):
    """Measure direct bearing against wind FROM and goal-axis current TO."""
    radians = np.deg2rad(bearing_deg)
    current = float(observation.current.u * np.sin(radians) + observation.current.v * np.cos(radians))
    wind_angle = abs((bearing_deg - observation.wind.direction + 180.) % 360. - 180.)
    upwind = bool(observation.wind.speed > 0. and wind_angle <= NATIVE_CONFIG.no_go_angle_deg)
    opposing = bool(current < CURRENT_CUTOFF_MPS)
    regime = "both" if upwind and opposing else "upwind" if upwind else "opposing_current" if opposing else "none"
    return {"upwind": upwind, "opposing_current": opposing, "targeted": upwind or opposing,
            "wind_to_mark_angle_deg": float(wind_angle), "current_toward_mark_mps": current, "regime": regime}


@lru_cache(maxsize=900)
def launch_weather(block, start, latitude, longitude):
    _, archive = archives(block)
    return archive.observe(BoatState(latitude, longitude, 0., 0.), normalise_time(start))


def stable_seed(key):
    return int.from_bytes(sha256(key.encode("utf-8")).digest()[:8], "big")


def traffic_recipe(split, seed, replica):
    names = TRAFFIC_NAMES[split]
    base = names[(seed + replica) % len(names)]
    profile = deepcopy(manifest()["traffic_profiles"][base])
    if replica:
        # Variants reuse the same registered encounter types. Distinct clock
        # ages and offsets exercise the same receiver and unchanged guard.
        profile["delay_s"] += (5, 10, 15)[replica - 1]
        lead = profile["targets"][0]
        lead["along_m"] += (25., -30., 40.)[replica - 1]
        lead["starboard_m"] += (-20., 30., -30.)[replica - 1]
        if replica >= 2:
            profile["drop_windows_s"] = [[10, 30], *profile["drop_windows_s"]]
    return f"stage1_{base}_v{replica}", profile


def make_roster(split, *, expansion=0):
    """Return cumulative deterministic launch specs, filtered only on past inputs.

    The baseline grid uses hourly model-valid clocks, two GPX source legs,
    modest rotations, one assigned AIS recipe and one initial trim per
    geometry/time cell. Expansion appends geometry and receiver variants;
    earlier IDs and specifications stay unchanged.
    """
    if split not in SPLITS or type(expansion) is not int or not 0 <= expansion <= MAX_EXPANSION:
        raise ValueError(f"Choose split from {SPLITS} and expansion in [0, {MAX_EXPANSION}].")
    geometries = geometry_variants(expansion)
    registered = manifest()
    rows = []
    for block in WEEKS[split]:
        archive = registered["archive_blocks"][block]
        first = normalise_time(archive["start_utc"])
        end = normalise_time(archive["end_utc_exclusive"])
        for time in np.arange(first, end - np.timedelta64(CAP_S, "s"), np.timedelta64(1, "h")):
            start = f'{time.astype("datetime64[s]")}Z'
            stamp = f'{time.astype("datetime64[m]")}'.replace("-", "").replace(":", "")
            clock = time.astype("datetime64[h]")
            window = clock.astype(np.int64) // 6 * 6
            weather_group = f'{block}__{np.datetime64(int(window), "h")}'
            for geometry in geometries:
                obs = launch_weather(block, start, *coordinates(geometry["departure"]))
                condition = conditions(obs, geometry["bearing_deg"])
                if not condition["targeted"]:
                    continue
                key = f'{split}__{block}__{stamp}Z__{geometry["route_id"]}'
                seed = stable_seed(key)
                for replica in range(expansion + 1):
                    profile, recipe = traffic_recipe(split, seed, replica)
                    heading_offset, surge, flap = INITIAL_TRIMS[(seed + replica) % len(INITIAL_TRIMS)]
                    rows.append({
                        "id": f"{key}__{profile}", "split": split, "archive_block": block,
                        "departure_utc": start, "cluster": weather_group, **geometry,
                        "initial_heading_deg": (geometry["bearing_deg"] + heading_offset) % 360.,
                        "initial_heading_offset_deg": heading_offset, "initial_surge_mps": surge,
                        "initial_flap_deg": flap, "traffic_profile": profile, "traffic": recipe,
                        "capture_radius_m": NATIVE_CONFIG.capture_radius_m, "control_dt_s": NATIVE_CONFIG.control_dt_s,
                        "max_steps": int(CAP_S / NATIVE_CONFIG.control_dt_s),
                        "geometry": condition, "expansion_replica": replica,
                    })
    return rows


@lru_cache(maxsize=1)
def engine_template():
    # The integrated Fossen plant and immutable CasADi mapaccum kernels are
    # shared by serial jobs inside each worker. Episode data are kept local.
    return CausalEfficientNativeEngine(None, config=NATIVE_CONFIG)


def scenario_parts(spec):
    """Reconstruct the identical analytic state, archive and held AIS receiver."""
    block, when = spec["archive_block"], normalise_time(spec["departure_utc"])
    plant, archive = archives(block)
    route = {"route_gpx": spec["route_gpx"], "waypoints": [spec["departure"], spec["goal"]],
             "first_leg_bearing_deg": spec["bearing_deg"]}
    receiver = ScenarioEnvironment(plant, spec, route, spec["traffic"])
    if receiver.max_target_acceleration_mps2 > NATIVE_CONFIG.acceleration_bound_mps2 + 1e-12:
        raise ValueError(f'{spec["id"]}: target acceleration exceeds the unchanged guard.')
    engine = copy(engine_template())
    engine.environment = CausalScenarioEnvironment(receiver, as_of_archive=archive)
    engine.anticipation_events = []

    latitude, longitude = coordinates(spec["departure"])
    heading, surge = spec["initial_heading_deg"], spec["initial_surge_mps"]
    initial = launch_weather(block, spec["departure_utc"], latitude, longitude)
    direction = np.deg2rad(heading)
    ground_speed = float(np.hypot(initial.current.u + surge * np.sin(direction),
                                  initial.current.v + surge * np.cos(direction)))
    state = BoatState(latitude, longitude, heading, ground_speed, flap=spec["initial_flap_deg"],
                      surge_speed=surge, sway_speed=0., yaw_rate=0.)
    dest = spec["goal"]
    goals = (RouteWaypoint(dest["latitude"], dest["longitude"], dest["waypoint"]),)
    return engine, state, goals, when, spec["max_steps"] * spec["control_dt_s"]


def prepare(spec):
    """Reconstruct one decision and its incumbent-first close, legal candidates."""
    engine, state, goals, start, cap = scenario_parts(spec)
    decision = engine.decision(state, goals[0], start, 0.)
    observed = conditions(decision.observation, spec["bearing_deg"])
    if observed != spec["geometry"]:
        raise ValueError(f'{spec["id"]}: causal geometry changed after registration.')
    eligible = gate_indices(decision, forecasts(decision))
    indices = []
    if observed["targeted"] and len(eligible) >= 1 + MIN_ALTERNATIVES:
        indices = [int(decision.physics_index), *(int(i) for i in eligible if i != decision.physics_index)]
    snapshot = NativeSnapshot(state, goals, start, 0., cap)
    return engine, snapshot, decision, indices


def discover(spec):
    """Screen the native launch without any branch continuation or outcomes."""
    _, _, decision, indices = prepare(spec)
    nominal = decision.physics_index
    eta = forecasts(decision)
    candidates = gate_indices(decision, eta)
    if indices:
        status = "eligible"
    elif nominal < 0:
        status = "no_safe_action"
    elif not np.isfinite(eta[nominal]):
        status = "physics_capture_not_predicted"
    else:
        status = "insufficient_eligible_alternatives"
    geometry = spec["geometry"]
    return {
        "id": spec["id"], "split": spec["split"], "cluster": spec["cluster"],
        "archive_block": spec["archive_block"], "route_id": spec["route_id"],
        "traffic_profile": spec["traffic_profile"], "departure_utc": spec["departure_utc"],
        "status": status, "upwind": geometry["upwind"], "opposing_current": geometry["opposing_current"],
        "targeted": geometry["targeted"], "regime": geometry["regime"],
        "screened_candidate_count": int(np.count_nonzero(decision.allowed)),
        "near_candidate_count": len(candidates),
        "eligible_indices": indices, "eligible_choice_ids": [decision.choices[i].id for i in indices],
        "forecast_eta_s": [float(eta[i]) for i in indices], "branches": len(indices),
    }

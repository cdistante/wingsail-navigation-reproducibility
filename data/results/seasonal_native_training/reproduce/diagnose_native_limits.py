"""Read-only initial persistent-forcing limits; never used to select splits."""
from __future__ import annotations

from argparse import ArgumentParser
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linprog

from native_branches import NativeEngine, _PersistentPlanner
from scenario_environment import ScenarioEnvironment
from seasonal_environment import SeasonalArchive
from world._geo import local_east_north_metres
from world.boat import BoatState
from world.scenario import RouteWaypoint


def main():
    study = Path(__file__).resolve().parents[1]
    parser = ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=study / "manifest_expanded.json")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    archives, rows = {}, []
    for case in manifest["scenarios"]:
        block = case["archive_block"]
        if block not in archives:
            archives[block] = SeasonalArchive(study / "forcing" / block)
        environment = ScenarioEnvironment.from_manifest(archives[block], manifest, case)
        start, target = manifest["routes"][case["route_id"]]["waypoints"][:2]
        state = BoatState(start["latitude"], start["longitude"], case["initial_heading_deg"],
                          0., flap=-10., surge_speed=0.)
        observation = environment.observe(state, case["departure_utc"])
        engine = NativeEngine(environment)
        planner = _PersistentPlanner(observation, engine.dynamics, engine.config)
        velocity, _, drift = planner._polar(observation)
        velocity = np.vstack((velocity[np.isfinite(velocity).all(axis=1)], drift))
        goal = np.array(local_east_north_metres(target["latitude"], target["longitude"],
                                               state.latitude, state.longitude))
        distance = float(np.linalg.norm(goal))
        along = velocity @ (goal / distance)
        across = velocity @ np.array((goal[1], -goal[0])) / distance
        tangent = 50. / np.sqrt(distance**2 - 50.**2)
        # A positive convex-mixture velocity must point inside the gate cone.
        fit = linprog(-along, A_ub=np.stack((across - tangent * along,
            -across - tangent * along, -along)), b_ub=[0., 0., 0.],
            A_eq=np.ones((1, len(velocity))), b_eq=[1.], bounds=(0., None), method="highs")
        decision = engine.decision(state, RouteWaypoint(target["latitude"], target["longitude"],
            target["waypoint"]), case["departure_utc"])
        rows.append({"scenario_id": case["id"], "split": case["split"],
            "wind_speed_mps": observation.wind.speed, "wind_from_deg": observation.wind.direction,
            "current_e_mps": observation.current.u, "current_n_mps": observation.current.v,
            "wave_height_m": observation.wave.height, "goal_distance_m": distance,
            "native_steady_gate_cone_feasible": bool(fit.success and -fit.fun > 1e-7),
            "native_steady_gate_cone_max_along_mps": float(-fit.fun) if fit.success else None,
            "forecast_capturing_candidates": int(sum(c.predicted_eta_s is not None for c in decision.choices)),
            "forecast_legal_capturing_candidates": int(sum(c.predicted_eta_s is not None and decision.allowed[i]
                for i, c in enumerate(decision.choices))),
            "best_closest_m": min((c.predicted_closest_m for c in decision.choices), default=None)})
    output = {"scope": "Present forcing only; convex mixtures of native steady Fossen modes plus passive current. "
        "The direction cone intersects the 50 m gate. This test does not prove future archive or native transient "
        "infeasibility, and does not filter the frozen scenario roster.",
        "manifest": str(args.manifest), "rows": rows, "cases": len(rows),
        "steady_cone_feasible": sum(r["native_steady_gate_cone_feasible"] for r in rows),
        "with_legal_capturing_forecast": sum(r["forecast_legal_capturing_candidates"] > 0 for r in rows)}
    path = study / "native_initial_physics_diagnostic.json"
    path.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: value for key, value in output.items() if key != "rows"}))


if __name__ == "__main__":
    main()

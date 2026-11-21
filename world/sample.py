"""Run a small end-to-end sample of the available maritime world layers.

Execute from the project root:

    python -m world.sample --time 2026-07-26T21:00:00Z
"""

from __future__ import annotations

import argparse

from world import (
    BoatAction,
    BoatState,
    Coastline,
    Currents,
    Fossen3DOFSaildroneDynamics,
    MaritimeWorld,
    OccupancyGrid,
    SaildroneDynamics,
    SaildroneParameters,
    Waves,
    Weather,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample available maritime environment layers.")
    parser.add_argument("--latitude", type=float, default=39.88)
    parser.add_argument("--longitude", type=float, default=18.42)
    parser.add_argument("--time", default="2026-07-26T21:00:00Z")
    parser.add_argument("--heading", type=float, default=90.0)
    parser.add_argument("--rudder", type=float, default=0.0)
    parser.add_argument("--flap", type=float, default=-10.0)
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument(
        "--model",
        choices=("fossen_3dof", "quasi_steady"),
        default="fossen_3dof",
        help="Dynamics model; legacy mode overrides this selection.",
    )
    parser.add_argument(
        "--legacy",
        action="store_true",
        help="Use the original simulator.py low-level dynamics and visualisation scale.",
    )
    arguments = parser.parse_args()

    world = MaritimeWorld(
        weather=Weather(),
        waves=Waves(),
        currents=Currents(),
        coastline=Coastline(),
    ).load()

    boat = BoatState(
        latitude=arguments.latitude,
        longitude=arguments.longitude,
        heading=arguments.heading,
        speed=0.0,
        flap=arguments.flap,
    )
    if arguments.legacy:
        dynamics = SaildroneDynamics(SaildroneParameters.legacy_visualisation())
    elif arguments.model == "fossen_3dof":
        dynamics = Fossen3DOFSaildroneDynamics()
    else:
        dynamics = SaildroneDynamics()
    transition = world.step(
        dynamics,
        boat,
        BoatAction(rudder=arguments.rudder, flap=arguments.flap),
        arguments.time,
        arguments.dt,
        grid_builder=OccupancyGrid(),
    )

    print(transition.observation)
    print(f"Next boat state: {transition.next_observation.boat}")
    print(f"Occupancy grid: {transition.grid.data.shape}, channels={transition.grid.channels}")
    world.close()


if __name__ == "__main__":
    main()

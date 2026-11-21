# Otranto GPX traffic stress check at 5°

This record describes the original experiment workspace. The [reviewer evidence index](../../../../docs/RESULTS.md) lists the released files and verification commands.

Both runs loaded the saved Otranto 001–005 GPX, started at waypoint 001 and aimed for 002 with a 50 m radius. The 25-minute cap covered the synthetic encounter and recovery of route guidance. Both used Gamma, Direct acados, forecast hold tacks, time-aware AIS and the COLREG course preference. The Transformer arm also loaded the frozen hold/release checkpoint with two-window release confirmation.

The archived start observation is 30 September 2026 at 02:59:20 UTC. The legacy fixture repeated `--time` in the command, and the runner used the final value. The saved replay and audit use that effective launch time. Synthetic ships turn and change speed around the two drop windows, with 10 s delivery delay and up to 30 s report age. The controlled wind comes from 5° at an archived initial speed of 5.04 m/s.

| Method | Minimum margin beyond nominal 25 m | Minimum full uncertainty margin around true ship tracks | Minimum full margin around AIS-projected tracks | Waypoint 002 at cap |
|---|---:|---:|---:|---:|
| Classical | +14.81 m | −1.08 m | −5.10 m | 514.90 m away |
| Transformer | +14.81 m | −1.08 m | −5.10 m | 494.20 m away |

An independent 0.05 s sweep found one action inside the full truth-centered uncertainty buffer in each run. Three actions entered the report-centered buffer. All report-centered entries occurred during the minimum-risk fallback and while AIS had priority. The 25 m hull clearance stayed intact. Including the extra 3 m steering reserve gives a minimum truth-centered margin of −4.08 m. At the three affected steps, none of the sampled fallback commands could maintain the full report-centered buffer. The weather model resumed on the route after the encounter. Both runs ended at the 25-minute cap on the first leg. All 301 frames in each labeled MP4 passed decode.

- Classical `replay`, `MP4`, `GPX`, and `command`
- Transformer `replay`, `MP4`, `GPX`, and `command`
- [Independent clearance audit](independent_traffic_sweep.json), `sampled fallback commands`, and `audit script`

# Seasonal native tack-selection study

This study compares physics-based tack selection with a learned correction to the same schedule forecasts. Matched native simulations test whether corrections fitted on Otranto improve ordered waypoint capture and elapsed time on the held-out Brindisi route family under autumn weather and new AIS encounter patterns.

## Environmental data and split

Each block supplies a seven-day UTC study window with 24 hours of preceding data and 48 hours of follow-through. The crop covers 39.85–40.85° N and 17.65–19.15° E. The hourly fields combine ERA5 10 m wind reanalysis, SMOC total surface current, and ECMWF WAM significant wave height, mean direction and spectral peak period. Wind and wave bearings use FROM. Current bearings use TO. The archive provides valid times. The schedule model uses persistence of the observed forcing at the decision time. The simulation advances through the archived fields. Raw responses, masks, source coordinates and hashes are recorded with the 12 NetCDFs. [Acquisition and source details](forcing/README.md) · [Archive index](forcing/acquisition_index.json) · [Coverage verification](forcing_validation.json)

All four seasonal blocks use the same products. Earlier V2 experiments drew wind from GFS, currents from HYCOM, and waves from Copernicus Marine. SMOC includes Eulerian current, wave drift, and tides. The Fossen plant uses that surface vector for advection and WAM significant wave height for damping. The other WAM fields remain in the archive for analysis. Each physics–hybrid pair uses identical inputs. [Earlier V2 data requirements](../../../README.md) · [SMOC product](https://doi.org/10.48670/moi-00016)

| Split | Seven-day windows in 2026 | Routes and synthetic AIS profiles | Scenarios |
|---|---|---|---:|
| Training | Jan 12–18 and Jul 13–19 | Two short Otranto GPXs with crossing and opposed traffic | 16 |
| Validation | Apr 13–19 | Same Otranto family with a slanted crossing profile | 6 |
| Held-out test | Oct 12–18 | Three Brindisi GPXs with oblique crossing and overtaking profiles | 12 |

The expanded roster fixes departures, route order, and traffic recipes before fitting. The 12 test scenarios form six route–departure groups, each crossed with two traffic profiles. Synthetic vessels use the first GPX leg as the coordinate frame. The two test profiles include 20 s AIS reception delay, two emission gaps, and a finite course/speed ramp. The test changes geography, the weather block, and traffic together, so it measures transfer under a joint distribution shift. [Registered roster](manifest_expanded.json) · [Independent input audit](independent_seasonal_inputs.json)

## Native episodes and matched training labels

Both methods use the same Fossen three-degree-of-freedom plant with 0.05 s integration, 5 s helm updates, slew-limited rudder and flap tracking, and an up-to-120 s tack commitment. A 48-minute persistence forecast ranks native schedules that sail on one trim or switch once. The chart screen uses the GEBCO depth grid. The report-based AIS and COLREG checks screen each proposed control and can interrupt a commitment. When no safe native action is available, the episode stops and records that outcome.

The schedule forecast uses steady Fossen equilibria and an estimated turning delay. The outcome simulation integrates the full plant at each native step. This experiment uses the native trim-feedback helm. The preceding Otranto replay comparison used Gamma with Direct acados NMPC and its dedicated AIS fallback.

Every mission has a six-hour cap. It starts at the first GPX point and sails the listed outbound marks. Gate-entry time is interpolated along the 0.05 s integration path using a 50 m radius. For training, the physics policy supplies up to five decision states per scenario. The incumbent and eligible alternative first tacks branch from the same boat state, UTC and AIS history. A branch ends at the next mark or a stop, with an 80-minute maximum horizon. A branch that reaches the gate with positive report- and truth-centred margins receives its observed time. Other outcomes receive twice the branch cap. Fitting uses pairs in which both requested first trims were applied for a positive duration. [Native simulator](reproduce/native_branches.py) · [Training and evaluation runner](reproduce/run_seasonal.py)

## Learned correction and evaluation

The residual ranker fits the difference between realized and forecast relative arrival times for each matched pair. It uses a state-weighted ridge regression over native boat and schedule features, delivered AIS features, and the frozen 48-value temporal Transformer embedding. The encoder takes 12 five-minute samples of wind and current vectors and significant wave height. Interactions with each candidate's relative bow let the same history affect candidate scores differently.

The hybrid considers admitted candidates with a predicted gate entry and at most 150 s of extra native forecast time over the incumbent. It limits each learned correction to ±900 s and requires more than 25 s of corrected advantage before switching. Spring validation compares correction weights 0, 0.3, and 1 by clearance violations, completions, and penalized time in that order. The held-out comparison pairs physics with the best nonzero weight from validation. The runner reports the overall validation preference separately. [Ranker and gate](reproduce/seasonal_ranker.py)

Evaluation retains stopped and capped runs. The penalized time score assigns twice the mission cap to a noncompletion. Evaluation also measures ordered waypoint completion, elapsed time, minimum report- and truth-centred clearance, and complete control-decision time relative to 5 s. The clearance envelope includes hull size, a 25 m gap, report uncertainty from error, age and bounded contact acceleration, and a 3 m reserve. The offline truth audit uses the same envelope around the synthetic ship positions. A run stopped before applying its first control has no travelled-segment clearance measurement. [Registered metrics](manifest_expanded.json)

## Results

Eleven of the 16 training scenarios supplied eligible states. The runner generated 236 applied branches from 45 exact-state starts and fitted 191 incumbent–alternative pairs. Every labeled branch reached its next 50 m gate with positive full clearance. Three training scenarios timed out and two stopped without an admissible action. None of those five supplied a state with competing forecast captures. The fit therefore represents choices among viable schedules and provides limited evidence about recovery when the forecast has no viable schedule. [Training receipt](training_receipt.json) · [Matched branch examples](native_tack_examples.jsonl.gz) · [Learned coefficients](seasonal_hybrid.npz)

Spring validation produced five ordered completions and zero clearance violations in six cases at each weight. It selected classical physics at weight zero. Weight 0.3 was the best active correction and was fixed for the held-out comparison. It added 7.97 s per scenario to the mean validation score. [Validation selection](validation_selection.json) · [Validation episodes](validation_episodes.csv)

| Held-out metric | Physics | Hybrid with weight 0.3 |
|---|---:|---:|
| Completed routes with every mark in order | 5/12 | 5/12 |
| Individual captures in order | 12/28 | 12/28 |
| Median time among completed routes | 1:01:28 | 1:03:09 |
| Capped at six hours | 1 | 1 |
| Stopped without an admissible native action | 6 | 6 |
| Runs with a measured full clearance margin | 10/12 | 10/12 |
| Runs that entered the full uncertainty envelope | 1 | 1 |
| Smallest report-centred margin | +0.17 m | +0.17 m |
| Smallest truth-centred margin | −1.10 m | −1.10 m |
| Largest complete control-decision time | 0.092 s | 0.088 s |
| Control attempts at or above 5 s | 0 | 0 |

| Completed route and departure UTC | Traffic | Physics | Hybrid | Hybrid minus physics |
|---|---|---:|---:|---:|
| Brindisi 003, Oct 13 18:00 | Oblique crossing | 1:07:54 | 1:07:54 | 0 s |
| Brindisi 003, Oct 14 12:00 | Oblique crossing | 0:59:47 | 1:02:00 | +133.05 s |
| Brindisi 003, Oct 14 12:00 | Raking overtake | 1:01:28 | 1:03:09 | +101.58 s |
| Brindisi 004, Oct 16 18:00 | Oblique crossing | 0:52:17 | 0:52:17 | 0 s |
| Brindisi 004, Oct 17 12:00 | Oblique crossing | 1:17:26 | 1:14:05 | −200.52 s |

The hybrid changed six tactical decisions across the held-out cases. It saved 3 min 21 s on one route, took longer on two, and tied on two. The median paired change was 0 s, and the hybrid added 6.82 s on average among the five jointly completed routes. Six route–departure groups and a single held-out week limit the generality of this estimate. Both the validation result and the held-out comparison support keeping physics as the default for this controller. [All paired episodes and captures](heldout_episodes.csv) · [Aggregate result](heldout_results.json)

The two Oct 18 Brindisi Small runs per arm stopped before applying a control. In the Oct 13 Brindisi 003 raking overtake, both methods took the same action during elapsed 165–170 s with 45 s old reports. The report-centred audit cleared by 0.166 m. The truth-centred full envelope was entered by 1.096 m, and the next command was blocked. This is a shared guard failure under stale reporting. The six-hour stop and absence of an admissible action remain distinct from completion. [Independent checks of 24 replays](independent_seasonal_heldout.json) · [Replay telemetry](telemetry)

## Reproduce the comparison

Run from the V2 root in the simulator Python environment with CasADi, PyTorch, NumPy, SciPy, xarray and netCDF4.

```sh
study=data/results/seasonal_native_training
export PYTHONPATH="$PWD:$PWD/$study/reproduce:$PWD/data/results/learned_weather_ais_policy/reproduce:$PWD/data/results/learned_safe_mark_policy/reproduce${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
python -B "$study/reproduce/run_seasonal.py" train --workers 8
python -B "$study/reproduce/run_seasonal.py" validate --workers 8
python -B "$study/reproduce/run_seasonal.py" test --workers 8
python -B "$study/reproduce/verify_seasonal_heldout.py"
```

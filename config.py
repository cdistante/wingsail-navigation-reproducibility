from pathlib import Path

ROOT = Path(__file__).parent

DATA = ROOT / "data"

WEATHER = DATA / "weather"

WAVES = DATA / "waves"

CURRENTS = DATA / "currents"

# HYCOM grid cells adjacent to the shoreline are often land-masked.  A normal
# bilinear interpolation then returns NaN even when a valid ocean cell exists
# a short distance offshore.  The current layer may use that nearest finite
# sea value, but never beyond this bounded distance.  Set this to zero to
# disable the coastal fallback entirely.
CURRENT_SAMPLING = {
    "coastal_fallback_max_distance_metres": 5_000.0,
}

BATHYMETRY = DATA / "bathymetry"

AIS = DATA / "ais"

AIS.mkdir(parents=True, exist_ok=True)

AIS_REPLAY = AIS / "replay"
AIS_REPLAY.mkdir(parents=True, exist_ok=True)

AIS_SYNTHETIC = AIS / "synthetic"
AIS_SYNTHETIC.mkdir(parents=True, exist_ok=True)

AIS_PROCESSED = AIS / "processed"
AIS_PROCESSED.mkdir(parents=True, exist_ok=True)

ENC = DATA / "enc"

COASTLINE = DATA / "coastline"

OCCUPANCY = DATA / "occupancy"

# Saildrone physical and control characteristics used by the default
# quasi-steady and Fossen 3-DOF models.  The values below were transferred
# from the supplied calculation results and parameter sheet (27 July 2026).
# They are SI unless a key explicitly states degrees.
#
# The source identifies the geometry as STL-derived and several aerodynamic,
# actuator, and hydrodynamic values as engineering estimates.  Consequently,
# added mass, sway/yaw damping, wing longitudinal position, and rudder drag
# remain explicitly provisional until identified from experiments or CFD.
SAILDRONE = {
    # Mass and wing sail
    "mass_kg": 52.2,
    "wing_sail_area_m2": 1.203,
    "air_density_kg_m3": 1.225,
    "centre_of_effort_height_m": 1.357,

    # Wing-sail / flap aerodynamics
    # This is the direct-lift approximation.  The separate tail-vane data is
    # retained below for a future explicitly resolved tail model.
    "flap_effectiveness": 0.25,
    "lift_curve_slope_per_rad": 3.494557591666905,
    "max_lift_coefficient": 1.00,
    "profile_drag_coefficient": 0.030,
    "induced_drag_factor": 0.12700431277058982,

    # Hull, keel, and sea-state response
    "hull_drag_coefficient_n_per_mps2": 9.0,
    "wave_drag_increase_per_metre": 0.35,
    "righting_moment_per_degree_nm": 2.30,
    "max_heel_degrees": 30.0,

    # Fossen 3-DOF surge/sway/yaw model.  The supplied resistance table below
    # is used for surge damping.  The fallback 9 u|u| resistance is retained
    # for custom parameter sets without the table.  Added mass, yaw inertia,
    # and sway/yaw damping have not been identified by the supplied files.
    # Added-mass entries are positive magnitudes added to rigid-body inertia.
    "yaw_inertia_kg_m2": 50.0,
    "added_mass_surge_kg": 15.0,
    "added_mass_sway_kg": 50.0,
    "added_yaw_inertia_kg_m2": 20.0,
    "linear_surge_damping_n_per_mps": 0.0,
    "quadratic_surge_damping_n_per_mps2": 9.0,
    "linear_sway_damping_n_per_mps": 80.0,
    "quadratic_sway_damping_n_per_mps2": 180.0,
    "linear_yaw_damping_nm_per_radps": 20.0,
    "quadratic_yaw_damping_nm_per_radps2": 60.0,
    "water_density_kg_m3": 1025.0,

    # Fossen control loads.  A negative longitudinal coordinate is aft of
    # the centre-of-buoyancy/3-DOF reference origin.  Positive rudder
    # commands follow the simulator's historic convention and turn to port.
    "rudder_area_m2": 0.08442925917762516,
    "rudder_lift_slope_per_rad": 2.84505091180687,
    "rudder_max_lift_coefficient": 1.10,
    "rudder_profile_drag_coefficient": 0.08,
    "rudder_induced_drag_factor": 0.12,
    "rudder_longitudinal_position_m": -1.0211504825635123,
    # No wing pivot longitudinal coordinate relative to the 3-DOF reference
    # origin is present in the supplied data; identify this from the layout.
    "wing_sail_longitudinal_position_m": 0.50,
    "fossen_max_integration_step_s": 0.05,

    # Actuator limits and low-order yaw response.  Rudder response is the
    # heading-rate coefficient in deg / (s · rudder_deg · m/s).
    "rudder_response_deg_per_s_per_deg_per_mps": 0.18,
    "rudder_min_degrees": -30.0,
    "rudder_max_degrees": 30.0,
    "flap_min_degrees": -15.0,
    "flap_max_degrees": 15.0,
    "flap_rate_degrees_per_s": 100.0,
}


# Total calm-water resistance envelope from saildrone_calculation_results.json.
# Each pair is (through-water speed [m/s], resistance [N]).  The Fossen
# dynamics interpolates the table inside this range and keeps a quadratic
# continuation outside it.  It is also a useful target for tow/coast-down
# validation.
SAILDRONE_SURGE_RESISTANCE_TABLE_N = (
    (0.50, 1.4562363498491395),
    (1.00, 5.995346175248446),
    (1.25, 11.137855222317189),
    (1.50, 21.29642912290661),
    (1.75, 40.26879198786445),
    (2.00, 70.40682392146124),
)


# STL-derived geometry and hydrostatics, retained in configuration even where
# the current planar model does not yet consume a value.  This makes the
# source data available for the future explicit tail, keel, added-mass and
# 6-DOF extensions without re-entering it by hand.
SAILDRONE_GEOMETRY_AND_HYDROSTATICS = {
    "water_density_kg_m3": 1025.0,
    "displacement_volume_m3": 0.050926829268292687,
    "hull_length_overall_m": 2.6284446792602543,
    "hull_beam_max_m": 0.6697053570747376,
    "length_waterline_m": 2.4222599960610682,
    "beam_waterline_m": 0.4718364982937473,
    "draft_total_m": 1.0346905406173192,
    "waterplane_area_m2": 0.7924175758171401,
    "waterplane_transverse_inertia_m4": 0.010301220363555005,
    "wetted_hull_area_m2": 1.539778442247676,
    "wetted_rudder_area_m2": 0.19086370020616272,
    "wetted_total_area_m2": 1.7306421424538385,
    "centre_buoyancy_x_m": 1.179927828108493,
    "centre_buoyancy_depth_below_wl_m": 0.1168858928191785,
    "BM_transverse_m": 0.2022749209318751,
    "GM_nominal_m": 0.2574300789287489,
    "CG_depth_below_wl_nominal_m": 0.1720410508160523,
    "wing_span_m": 2.0050001068115235,
    "wing_chord_m": 0.599989990234375,
    "wing_max_thickness_m": 0.10734115600585942,
    "wing_thickness_ratio": 0.1789049113368184,
    "wing_aspect_ratio": 3.341722594452463,
    "wing_pivot_fraction_from_leading_edge": 0.25115419406004114,
    "rudder_span_m": 0.43163281249999974,
    "rudder_mean_chord_m": 0.19560435799265194,
    "rudder_aspect_ratio": 2.206662555627797,
    "rudder_arm_from_centre_buoyancy_m": 1.0211504825635123,
}


# Separated tail-vane (called flap in the supplied geometry file).  The
# present models use the documented direct-lift flap approximation above;
# these values are intentionally not folded into a fictitious wing coefficient.
SAILDRONE_TAIL_VANE = {
    "area_m2": 0.12166767089843748,
    "span_m": 0.690,
    "mean_chord_m": 0.17632995782382246,
    "aspect_ratio": 3.91311838620981,
    "area_ratio_to_wing": 0.10113856544091866,
    "moment_arm_from_wing_pivot_m": 1.2995423987531463,
    "tail_volume_coefficient": 0.2190600777993001,
    "lift_slope_per_rad": 3.7367250984497082,
}


# Local tactical planner used to produce collision-aware heading/path references
# for the MPC.  It is deliberately separate from the vessel dynamics: the
# planner chooses a feasible track, while the MPC later chooses flap and rudder
# commands subject to the Fossen model and actuator limits.
HARMONIC_TACTICAL_PLANNER = {
    # Sailboat no-go cone, measured around the meteorological true-wind FROM
    # bearing.  A heading inside this cone is not allowed during path
    # extraction; set this from sea-trial/polar data when available.
    "no_go_angle_degrees": 45.0,
    # A moderate anisotropy biases the field away from the upwind cone without
    # overwhelming the geographic goal gradient.  Stronger values previously
    # allowed a local field to point backwards along an otherwise open route.
    "minimum_diffusion": 0.60,
    "transition_band_multiplier": 1.50,
    # Do not emit a local path segment with a negative projection on the
    # active route/waypoint direction.  A value of zero permits a lateral
    # safety manoeuvre but prevents a U-turn away from the mission goal.
    "minimum_goal_progress_cosine": 0.25,

    # Sparse BiCGSTAB solve for the directionally weighted harmonic field.
    "maximum_iterations": 1_000,
    "convergence_tolerance": 1.0e-6,
    "maximum_path_steps": 500,

    # AIS targets are projected at constant reported COG/SOG and painted as a
    # conservative swept obstacle.  These values are tactical-safety defaults,
    # not COLREGs or a certified collision-avoidance envelope.
    # This is a receding horizon: replan at each MPC update rather than
    # collapsing long-term moving traffic into one overly conservative map.
    "ais_prediction_horizon_seconds": 60.0,
    "ais_prediction_interval_seconds": 5.0,
    "ais_safety_radius_metres": 25.0,

    # Only apply the depth constraint when bathymetry is actually available in
    # the observation.  The current local bathymetry source is point-sampled.
    "minimum_depth_metres": 3.0,
}


# Conservative, AIS-informed encounter policy used by the optional COLREG
# layer.  This is a transparent good-visibility simulation policy: AIS alone
# cannot establish every fact required by the legal rules (lookout, vessel
# status, manoeuvrability, sail tack, visibility, and local traffic rules).
# It therefore adds a tactical safety bias to the local planner and must not
# be treated as a certified autonomous collision-avoidance system.
#
# Bearings are true degrees clockwise from north.  The crossing convention
# follows Rules 13--17 for ordinary power-driven-vessel encounters: an
# approaching vessel on the saildrone's starboard side makes the saildrone
# give way, with a substantial starboard manoeuvre preferred.  A sailing
# vessel's Rule-12 tack information is not available from standard AIS and is
# deliberately not inferred here.
COLREG_POLICY = {
    # Assess the constant-COG/SOG closest point of approach over this rolling
    # tactical horizon.  Reassessment happens at every harmonic replan.
    "assessment_horizon_seconds": 900.0,
    "risk_dcpa_metres": 100.0,
    "emergency_dcpa_metres": 40.0,
    "emergency_tcpa_seconds": 30.0,
    "minimum_relative_speed_metres_per_second": 0.05,

    # Encounter classification sectors.
    "head_on_bearing_degrees": 15.0,
    "head_on_course_tolerance_degrees": 30.0,
    "overtaking_sector_degrees": 112.5,
    "crossing_sector_degrees": 112.5,

    # For a give-way / head-on encounter, reserve a port-forward wedge in the
    # local harmonic grid.  This avoids a port alteration and leaves a
    # starboard manoeuvre as the preferred route around the AIS corridor.
    "starboard_turn_prohibited_sector_degrees": 100.0,
    "starboard_turn_clearance_metres": 250.0,
}


# Discrete nonlinear MPC settings.  Each candidate rudder/flap pair is held
# over the short prediction horizon and rolled out through the existing Fossen
# model.  This is a deterministic direct-shooting baseline, deliberately kept
# small enough for local replanning before a continuous optimiser is added.
SAILDRONE_MPC = {
    "prediction_horizon_steps": 8,
    "prediction_dt_seconds": 1.0,
    "rudder_candidates_degrees": (-30.0, -22.5, -15.0, -7.5, 0.0, 7.5, 15.0, 22.5, 30.0),
    "flap_candidates_degrees": (-15.0, -10.0, -5.0, 0.0, 5.0, 10.0, 15.0),
    "close_hauled_rudder_step_degrees": 1.0,
    "heading_error_weight": 6.0,
    "progress_reward_per_metre": 0.25,
    # Reject a rollout that makes ground-track progress away from the active
    # harmonic target.  This complements (rather than replaces) the local
    # obstacle field and prevents an MPC turn from becoming a route U-turn.
    "reverse_route_progress_penalty_per_metre": 10.0,
    "harmonic_potential_weight": 5.0,
    "obstacle_penalty": 100_000.0,
    "outside_grid_penalty": 25.0,
    "no_go_penalty": 50.0,
    # Heel is a soft safety constraint, not a reason to abandon a waypoint
    # correction for a few degrees of normal sailing heel.
    "heel_soft_limit_degrees": 10.0,
    "heel_weight": 0.05,
    "control_effort_weight": 0.10,
    "control_change_weight": 0.20,
    # Command slew bounds at the one-second MPC update rate.  The Fossen
    # dynamics presently models the rudder as instantaneous, so these are
    # controller-side actuator limits that suppress command chatter until a
    # measured rudder servo model is available.
    "max_rudder_change_per_update_degrees": 7.5,
    "max_flap_change_per_update_degrees": 5.0,
}

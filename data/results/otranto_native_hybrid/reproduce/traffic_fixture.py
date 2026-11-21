"""Study-local analytical traffic parameters, delivered through the native AIS receiver.

The simulator reads only the parameter JSON. The independently exported truth
CSV is never opened here or passed to the navigation controller.
"""
from __future__ import annotations

from dataclasses import fields
from hashlib import sha256
import json
import math
from pathlib import Path

from world.synthetic_ais import SyntheticAIS, SyntheticAISManoeuvre, SyntheticAISTarget


def fixture_provider(path, *, expected_contacts=3):
    path=Path(path).resolve()
    config=json.loads(path.read_text())
    origin=tuple(float(value) for value in config['origin_latitude_longitude'])
    if len(origin)!=2 or not all(math.isfinite(v) for v in origin):
        raise ValueError('Traffic fixture needs one finite latitude/longitude origin.')
    allowed={field.name for field in fields(SyntheticAISTarget)}
    records=config['scenarios']['mixed']
    if len(records) != expected_contacts:
        raise ValueError(f'The fixture requires exactly {expected_contacts} contacts.')
    if any(set(row)-allowed for row in records):
        raise ValueError('Unexpected synthetic target fields in fixture parameters.')
    targets=tuple(SyntheticAISTarget(**row) for row in records)
    identities={target.mmsi for target in targets}
    if len(identities)!=len(targets):raise ValueError('Traffic MMSIs must be unique.')
    manoeuvres={}
    for row in config['velocity_ramps']:
        identifier=int(row['mmsi'])
        if identifier not in identities or identifier in manoeuvres:
            raise ValueError('Each maneuver must identify one unique fixture contact.')
        values=[float(row[key]) for key in ('start_s','end_s','final_course_degrees','final_speed_mps')]
        start,stop,course,speed=values
        if not all(math.isfinite(v) for v in values) or stop<=start or start<0 or speed<0:
            raise ValueError('Invalid analytical traffic ramp.')
        manoeuvres[identifier]=SyntheticAISManoeuvre(start,stop,course,speed)

    class FixtureAIS(SyntheticAIS):
        _SCENARIOS={'mixed':targets}

        def __init__(self,scenario,origin_latitude,origin_longitude,**kwargs):
            if abs(origin_latitude-origin[0])>1e-9 or abs(origin_longitude-origin[1])>1e-9:
                raise ValueError('Fixture origin differs from the vessel GPX departure.')
            super().__init__(scenario,origin_latitude,origin_longitude,**kwargs)
            self._manoeuvres=manoeuvres
            self.manoeuvring=bool(manoeuvres)

    return FixtureAIS,config,sha256(path.read_bytes()).hexdigest()


def with_underway_contact(initial_provider, later_provider, first_emission_seconds):
    """Add a newly transmitting contact on the existing emission/delivery clock."""
    if not math.isfinite(first_emission_seconds) or first_emission_seconds < 0:
        raise ValueError("The first underway-contact emission must have a valid epoch.")
    initial_ids = {target.mmsi for target in initial_provider._SCENARIOS["mixed"]}
    if any(target.mmsi in initial_ids for target in later_provider._SCENARIOS["mixed"]):
        raise ValueError("The initial and underway fixtures must have distinct contact identities.")

    class CombinedAIS(initial_provider):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            if self.reporting is None:
                raise ValueError("The underway study requires a delayed reporting receiver.")
            self.underway = later_provider(*args, **kwargs)

        def load(self):
            super().load()
            self.underway.load()
            return self

        def close(self):
            self.underway.close()
            super().close()

        def targets(self, when=None):
            reports = super().targets(when)
            elapsed = 0.0 if when is None else float(when)
            generation = self.reporting.latest_epoch(elapsed - self.launch_lead_seconds)
            if generation >= first_emission_seconds:
                reports.extend(self.underway.targets(elapsed))
            return reports

    return CombinedAIS

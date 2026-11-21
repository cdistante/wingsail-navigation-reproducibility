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


def fixture_provider(path):
    path=Path(path).resolve()
    config=json.loads(path.read_text())
    origin=tuple(float(value) for value in config['origin_latitude_longitude'])
    if len(origin)!=2 or not all(math.isfinite(v) for v in origin):
        raise ValueError('Traffic fixture needs one finite latitude/longitude origin.')
    allowed={field.name for field in fields(SyntheticAISTarget)}
    records=config['scenarios']['mixed']
    if len(records)!=3:
        raise ValueError('The new-encounter protocol requires exactly three contacts.')
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

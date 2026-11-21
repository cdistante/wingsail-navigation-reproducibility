"""Cross-check native report delivery against independently integrated fixture parameters."""
from __future__ import annotations
import hashlib,json
from pathlib import Path
import numpy as np
from define_encounters import sample
from traffic_fixture import fixture_provider
from world.synthetic_ais import SyntheticAISReporting
from world._geo import local_east_north_metres

STUDY=Path(__file__).resolve().parents[1]

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def generation(time):
 epoch=np.floor((time-20.)/5.)*5.
 while any(a<=epoch<b for a,b in ((85.,125.),(145.,185.))):epoch-=5.
 return epoch

def main():
 rows=[]
 for fixture in sorted((STUDY/'fixtures').glob('*/ground_truth.json')):
  Native,config,checksum=fixture_provider(fixture)
  origin=config['origin_latitude_longitude']
  receiver=Native('mixed',*origin,reporting=SyntheticAISReporting(5.,20.,((85.,125.),(145.,185.))),sharp_manoeuvres=True,launch_lead_seconds=40.).load()
  targets=config['scenarios']['mixed'];ramps={r['mmsi']:r for r in config['velocity_ramps']}
  times=sorted(set(np.arange(0,641,5.).tolist()+[185.-1e-6,185.,245.-1e-6,245.]))
  max_position=max_velocity=max_age_error=0.;ages=[]
  for mission_time in times:
   scenario_time=mission_time-40.;epoch=generation(scenario_time)
   reports={x.mmsi:x for x in receiver.targets(mission_time)}
   for target in targets:
    report=reports[target['mmsi']]
    position,velocity=sample(target,ramps[target['mmsi']],[epoch])
    actual=np.array(local_east_north_metres(report.latitude,report.longitude,*origin))
    angle=np.radians(report.course);reported_velocity=report.speed*np.array((np.sin(angle),np.cos(angle)))
    max_position=max(max_position,float(np.linalg.norm(actual-position[0])))
    max_velocity=max(max_velocity,float(np.linalg.norm(reported_velocity-velocity[0])))
    max_age_error=max(max_age_error,abs(report.report_age_seconds-(scenario_time-epoch)))
    ages.append(report.report_age_seconds)
  assert max_position<.001 and max_velocity<1e-8 and max_age_error<1e-8
  assert abs(max(ages)-(65.-1e-6))<1e-7
  rows.append({'encounter_id':fixture.parent.name,'fixture_sha256':checksum,'samples_per_vessel':len(times),'vessels':len(targets),
    'maximum_position_error_m':max_position,'maximum_velocity_error_mps':max_velocity,'maximum_age_error_s':max_age_error,
    'maximum_received_age_just_before_delivery_s':max(ages),'sampled5s_command_start_max_age_s':60.,'held_command_end_max_age_s':65.,
    'delivery_boundaries_mission_s':[185.,245.],'all_checks_passed':True})
  receiver.close()
 result={'all_passed':all(x['all_checks_passed'] for x in rows),'cases':rows,'selection_scope':'Input definition and receiver equivalence only. No controller actions, outcomes or optimized sequences are read.',
   'runtime_helper_sha256':digest(STUDY/'reproduce/traffic_fixture.py'),'independent_integrator_sha256':digest(STUDY/'reproduce/define_encounters.py'),
   'script_sha256':digest(Path(__file__))}
 (STUDY/'fixture_delivery_check.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps(result,indent=2))

if __name__=='__main__':main()

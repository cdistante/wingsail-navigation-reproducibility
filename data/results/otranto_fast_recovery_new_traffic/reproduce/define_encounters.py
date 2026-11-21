"""Predefine new traffic independently of controller outcomes and export analytical truth."""
from __future__ import annotations
import csv
import gzip
import hashlib
import json
from pathlib import Path
from xml.etree import ElementTree
import numpy as np

STUDY=Path(__file__).resolve().parents[1]
ROOT=STUDY.parents[2]
ORIGINAL=ROOT/'data/results/otranto_weak_wind_ais_fast_recovery'
PARAMETERS={
 'crossing_fan':[
  (998100201,'FERRY',60,60.,12.,20.,3.7,140.,100.,80.,85.,125.,32.,3.45),
  (998100202,'CARGO',70,75.,14.,255.,2.8,150.,20.,50.,145.,185.,242.,3.1),
  (998100203,'TUG',52,26.,9.,70.,3.2,130.,-90.,-15.,85.,125.,57.,3.5)],
 'opposed_crossing':[
  (998100301,'FERRY',60,60.,12.,335.,3.4,150.,-80.,145.,145.,185.,322.,3.8),
  (998100302,'CARGO',70,75.,14.,95.,2.95,155.,40.,60.,85.,125.,108.,2.65),
  (998100303,'TUG',52,26.,9.,250.,3.1,135.,110.,-15.,145.,185.,262.,2.7)]}


def velocity(course,speed):
 a=np.radians(course);return speed*np.array((np.sin(a),np.cos(a)))


def sample(target,ramp,times):
 times=np.asarray(times,dtype=float)
 old=velocity(target['course_degrees'],target['speed_metres_per_second'])
 new=velocity(ramp['final_course_degrees'],ramp['final_speed_mps'])
 duration=ramp['end_s']-ramp['start_s']
 turning=np.clip(times-ramp['start_s'],0.,duration)
 after=np.maximum(times-ramp['end_s'],0.)
 delta=new-old
 origin=np.array((target['initial_east_metres'],target['initial_north_metres']))
 position=origin+times[:,None]*old+(.5*turning**2/duration+after)[:,None]*delta
 speed=old+turning[:,None]/duration*delta
 return position,speed


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
 from config import SAILDRONE_GEOMETRY_AND_HYDROSTATICS as OWN
 route=ROOT/'current_route/otranto_short_test_loop.gpx'
 first=ElementTree.parse(route).find('.//{*}rtept')
 assert first.findtext('{*}name')=='001'
 origin=[float(first.attrib['lat']),float(first.attrib['lon'])]
 initial=ORIGINAL/'runs/otranto_355__classical/navigation.jsonl.gz'
 with gzip.open(initial,'rt') as stream:
  next(stream);observation=json.loads(next(stream))['observation']
 current=np.array([observation['current'][key] for key in ('u','v')])
 own_radius=.5*np.hypot(OWN['hull_length_overall_m'],OWN['hull_beam_max_m'])
 times=np.arange(7201)*.05
 preview_age=np.minimum(20.+times,65.)
 uncertainty=5.+np.maximum(.05*preview_age,.5*.026*preview_age**2)
 definition={'departure_utc':'2026-09-30T23:50:00Z','encounter_lead_s':40.,'report_interval_s':5.,'report_delay_s':20.,
  'drop_emission_windows_s':[[85.,125.],[145.,185.]],'maximum_held_report_age_s':65.,'preview_age_budget_s':65.,
  'age_peaks_encounter_s':[145.,205.],'age_peaks_mission_s':[185.,245.],
  'age_derivation':'Last emissions80/140 are retained until receipts125+20/185+20; applied intervals140–145/200–205 reach ages60–65 immediately before the new observation resets to20.',
  'acceleration_bound_mps2':.026,'nominal_gap_m':25.,'initial_uncertainty_m':5.,'linear_growth_mps':.05,'steering_reserve_m':3.,
  'own_initial_bow_deg':45.,'own_initial_speed_mps':0.,'own_initial_flap_deg':-10.,
  'fixture_selection':'Both predefined fleets retained regardless of subsequent controller outcome; no outcome-based tuning or exclusion.',
  'eligibility_definition':'Strictly positive initial truth/report full uncertainty plus reserve, acceleration vectors below.026, and at least one received-report CV full-envelope threat over360s to a passive-current CV ownship from zero water speed. This is input eligibility, not control feasibility.',
  'passive_initial_ground_velocity_mps':current.tolist(),'initial_current_source_replay_sha256':digest(initial),
  'initial_current_scope':'Only first observation current u/v read; no controls or trajectories used.',
  'route_sha256':digest(route),'source_sha256':digest(Path(__file__)),'cases':{}}
 for case,values in PARAMETERS.items():
  targets=[];ramps=[];anchors=[]
  for mmsi,name,kind,length,beam,course,speed,at,east,north,start,end,newcourse,newspeed in values:
   p=np.array((east,north))-at*velocity(course,speed)
   targets.append(dict(mmsi=mmsi,name=f'VALIDATION {case.upper()} {name}',callsign=f'VAL{mmsi%1000:03d}',ship_type=kind,
     initial_east_metres=float(p[0]),initial_north_metres=float(p[1]),speed_metres_per_second=speed,course_degrees=course,
     length_metres=length,beam_metres=beam,destination='OFFLINE VALIDATION',heading_degrees=None))
   acceleration=float(np.linalg.norm(velocity(newcourse,newspeed)-velocity(course,speed))/(end-start))
   assert acceleration<.026,(case,mmsi,acceleration)
   ramps.append(dict(mmsi=mmsi,start_s=start,end_s=end,final_course_degrees=newcourse,final_speed_mps=newspeed,
                     acceleration_magnitude_mps2=acceleration))
   anchors.append(dict(mmsi=mmsi,unmanoeuvred_anchor_s=at,east_m=east,north_m=north))
  checks=[]
  for target,ramp in zip(targets,ramps,strict=True):
   truth,initial_velocity=sample(target,ramp,[-40.])
   report,report_velocity=sample(target,ramp,[-60.])
   projected=report[0]+20.*report_velocity[0]
   hull=own_radius+.5*np.hypot(target['length_metres'],target['beam_metres'])
   radius=hull+25.+5.+max(.05*20,.5*.026*20**2)+3.
   truth_margin=float(np.linalg.norm(truth[0])-radius);report_margin=float(np.linalg.norm(projected)-radius)
   assert truth_margin>0 and report_margin>0
   centres=projected+times[:,None]*report_velocity[0]
   margin=np.linalg.norm(centres-times[:,None]*current,axis=1)-(hull+25.+uncertainty+3.)
   ix=int(np.argmin(margin))
   checks.append(dict(mmsi=target['mmsi'],initial_truth_full_reserve_m=truth_margin,initial_report_full_reserve_m=report_margin,
        passive_current_cv_preview_margin_m=float(margin[ix]),passive_cv_minimum_time_s=float(times[ix]),
        target_acceleration_mps2=ramp['acceleration_magnitude_mps2']))
  assert min(x['passive_current_cv_preview_margin_m'] for x in checks)<0,(case,'No avoidance-relevant input threat')
  folder=STUDY/'fixtures'/case;folder.mkdir(parents=True,exist_ok=True)
  fixture=folder/'ground_truth.json';trace=folder/'truth_trace.csv'
  configuration={'fixture':case,'uncertainty_note':'Synthetic emitted positions/velocities are exact; uncertainty models delayed/dropout propagation.',
    'position_model':'Independent analytic integration of piecewise linear EN velocity ramps; constant before and after each40s ramp.',
    'source':'Study-local predefined fixture parameters, no controller outcomes used.',
    'origin_latitude_longitude':origin,'scenarios':{'mixed':targets},'velocity_ramps':ramps,'design_anchors':anchors,
    'truth_trace':{'file':'truth_trace.csv','first_s':-80.,'last_s':600.,'step_s':.05},'reporting':{k:definition[k] for k in ('report_interval_s','report_delay_s','drop_emission_windows_s','maximum_held_report_age_s','preview_age_budget_s')}}
  if fixture.exists():assert json.loads(fixture.read_text())==configuration,'Frozen fixture cannot be overwritten by changed parameters.'
  else:fixture.write_text(json.dumps(configuration,indent=2)+'\n')
  timegrid=np.arange(-1600,12001)*.05
  if not trace.exists():
   with trace.open('w',newline='') as stream:
    writer=csv.writer(stream,lineterminator='\n');writer.writerow(['elapsed_s','mmsi','east_m','north_m','east_velocity_mps','north_velocity_mps'])
    for target,ramp in zip(targets,ramps,strict=True):
     p,v=sample(target,ramp,timegrid)
     writer.writerows((f'{t:.2f}',target['mmsi'],f'{pos[0]:.12f}',f'{pos[1]:.12f}',f'{vel[0]:.12f}',f'{vel[1]:.12f}') for t,pos,vel in zip(timegrid,p,v,strict=True))
  definition['cases'][case]={'fixture_sha256':digest(fixture),'truth_trace_sha256':digest(trace),'checks':checks,
       'initial_full_envelopes_strictly_clear':True,'passive_current_cv_threat_present':True,'all_accelerations_below_bound':True}
 output=STUDY/'fixture_eligibility.json'
 if output.exists():assert json.loads(output.read_text())==definition,'Eligibility record immutable.'
 else:output.write_text(json.dumps(definition,indent=2)+'\n')
 for case,row in definition['cases'].items():print(case,row['checks'])

if __name__=='__main__':main()

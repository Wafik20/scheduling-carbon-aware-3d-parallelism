import json
import pandas as pd
from opportunity_bounds import BASE,OUT,integral
s=pd.read_csv(OUT/'opportunity_sensitivity.csv')
for r in s.itertuples():
 w=json.loads((BASE/r.model/'workload.json').read_text()); m=pd.read_csv(w['profile']); ci=pd.read_csv(w['trace']).carbon.to_numpy(float)
 a=m.iloc[r.one_switch_first_action]; b=m.iloc[r.one_switch_second_action]; t=r.one_switch_at_h
 rem=w['tokens']-t*3600*a.throughput_tok_s; duration=rem/b.throughput_tok_s/3600
 carbon=(a.power_w*integral(ci,r.offset_h,t)+b.power_w*integral(ci,r.offset_h+t,duration)+400*integral(ci,r.offset_h+t+duration,b.save_s/3600))/1000
 assert abs(carbon-r.one_switch_carbon_g)<1e-5
 assert abs(t+duration+b.save_s/3600-r.one_switch_completion_h)<1e-8
 assert r.one_switch_completion_h<=r.deadline_h+1e-8
 assert r.lp_lower_bound_g<=r.one_switch_carbon_g+1e-5
print('PASS: all 90 one-switch schedules independently scalar-replayed; deadlines and LP bound ordering pass.')

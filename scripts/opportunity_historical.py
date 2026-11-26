"""Controlled changes using CURRENT measured GPT2 profile, not historical reproduction."""
import json
import pandas as pd
from opportunity_bounds import BASE,OUT, fixed,lower_bound,one_switch
w=json.loads((BASE/'gpt2/workload.json').read_text()); menu=pd.read_csv(w['profile']); ci=pd.read_csv(w['trace']).carbon.to_numpy(float)
cases=[('current',w['tokens'],9,0,1),('deadline_only_26h',w['tokens'],26,0,1),('tokens_only_8B',8e9,9,0,1),('historical_workload_current_profile',8e9,26,0,1),('historical_workload_offset6',8e9,26,6,1),('historical_workload_offset12',8e9,26,12,1),('historical_workload_offset18',8e9,26,18,1),('historical_workload_interval_quarter',8e9,26,0,.25),('historical_workload_interval4',8e9,26,0,4)]
rows=[]
for label,tokens,deadline,offset,interval in cases:
    f=fixed(menu,ci,tokens,deadline,offset)
    if f[1]<0:
        rows.append(dict(case=label,tokens=tokens,deadline_h=deadline,offset_h=offset,interval_h=interval,status='infeasible even fastest'))
        continue
    lb,_=lower_bound(menu,ci,tokens,deadline,offset); pair=one_switch(menu,ci,tokens,deadline,offset,interval)
    rows.append(dict(case=label,tokens=tokens,deadline_h=deadline,offset_h=offset,interval_h=interval,status='feasible',fixed_g=f[0],one_switch_g=pair['carbon_g'],one_switch_saving_pct=100*(f[0]-pair['carbon_g'])/f[0],lp_lower_bound_g=lb,**{k:v for k,v in pair.items() if k!='carbon_g'}))
pd.DataFrame(rows).to_csv(OUT/'opportunity_historical_control.csv',index=False)

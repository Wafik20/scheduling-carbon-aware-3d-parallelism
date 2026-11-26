"""Independent measured-menu opportunity bounds; does not import production solver."""
import json, math
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import coo_matrix
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/measured_models_independent_audit'
BASE=ROOT/'runs/measured_models_corrected_west1'
FINE=ROOT/'runs/measured_models_corrected_west1_fine'

def integral(ci, start, duration):
    total=0.
    end=start+duration
    while start < end-1e-10:
        edge=min(end, math.floor(start+1e-10)+1)
        total+=(edge-start)*ci[int(math.floor(start+1e-10))%len(ci)]
        start=edge
    return total

def fixed(menu, ci, tokens, deadline, offset=0):
    candidates=[]
    for idx,r in menu.iterrows():
        train=tokens/r.throughput_tok_s/3600
        duration=train+r.save_s/3600
        if duration<=deadline+1e-9:
            carbon=(r.power_w*integral(ci,offset,train)+400*integral(ci,offset+train,r.save_s/3600))/1000
            candidates.append((carbon,int(idx),duration))
    return min(candidates) if candidates else (float('nan'),-1,float('nan'))

def lower_bound(menu,ci,tokens,deadline,offset=0):
    # x[h,a] is fraction of hour spent training action a; arbitrary idle and mixing allowed.
    # Save is omitted: removing nonnegative energy/time constraints only enlarges feasible set.
    H=math.ceil(deadline); A=len(menu)
    duration=np.minimum(1.,deadline-np.arange(H))
    costs=np.array([integral(ci,offset+h,duration[h])/duration[h] for h in range(H)])[:,None]*menu.power_w.to_numpy()[None,:]/1000
    rates=np.tile(menu.throughput_tok_s.to_numpy()*3600/tokens,H)
    M=coo_matrix((np.ones(H*A),(np.repeat(np.arange(H),A),np.arange(H*A))),shape=(H,H*A)).tocsr()
    sol=linprog(costs.ravel(),A_ub=M,b_ub=duration,A_eq=rates[None,:],b_eq=[1.],bounds=(0,None),method='highs')
    if not sol.success: raise RuntimeError(sol.message)
    residual=abs(float(rates@sol.x)-1)
    return float(sol.fun),residual

def one_switch(menu, ci, tokens, deadline, offset=0, interval=1):
    """Enumerate every ordered action pair and grid-aligned single switch; no idle."""
    rate=menu.throughput_tok_s.to_numpy()*3600
    power=menu.power_w.to_numpy()/1000
    save=menu.save_s.to_numpy()/3600
    prefix=np.r_[0.,np.cumsum(ci)]; period=len(ci)
    def primitive(t):
        t=np.asarray(t); n=np.floor(t).astype(np.int64)
        return (n//period)*prefix[-1]+prefix[n%period]+(t-n)*ci[n%period]
    best=fixed(menu,ci,tokens,deadline,offset)
    answer=dict(carbon_g=best[0],first_action=best[1],second_action=best[1],switch_h=0.,completion_h=best[2])
    for t in np.arange(interval,deadline,interval):
        remaining=tokens-rate*t
        end=t+remaining[:,None]/rate[None,:]
        finish=end+save[None,:]
        valid=(remaining[:,None]>0)&(finish<=deadline+1e-9)
        carbon=power[:,None]*(primitive(offset+t)-primitive(offset))+power[None,:]*(primitive(offset+end)-primitive(offset+t))+.4*(primitive(offset+finish)-primitive(offset+end))
        carbon=np.where(valid,carbon,np.inf)
        idx=np.unravel_index(np.argmin(carbon),carbon.shape)
        if carbon[idx]<answer['carbon_g']-1e-9:
            answer=dict(carbon_g=float(carbon[idx]),first_action=int(idx[0]),second_action=int(idx[1]),switch_h=float(t),completion_h=float(finish[idx]))
    return answer

def run():
    metrics=[]; sensitivity=[]
    for wp in sorted(BASE.glob('*/workload.json')):
        w=json.loads(wp.read_text()); model=w['model']; menu=pd.read_csv(wp.parent/'strategy_profile.csv')
        ci=pd.read_csv(w['trace']).carbon.to_numpy(float)
        s=json.loads((wp.parent/'summary.json').read_text())
        fine_path=FINE/model/'summary.json'
        f=json.loads(fine_path.read_text()) if fine_path.exists() else s
        fine_record=next(c for c in json.loads((ROOT/'runs/measured_models_audit/candidates_recovered_qdiv10.json').read_text()) if c['model']==model)
        remaining=w['tokens']; fine_carbon=0.; cursor=0.
        for action in fine_record['actions']:
            row=menu.iloc[action]
            duration=min(1.,remaining/row.throughput_tok_s/3600)
            fine_carbon+=row.power_w*integral(ci,cursor,duration)/1000
            remaining-=duration*3600*row.throughput_tok_s; cursor+=duration
            if remaining<1e-4:
                fine_carbon+=400*integral(ci,cursor,row.save_s/3600)/1000
                cursor+=row.save_s/3600
                break
        assert remaining<1e-4 and cursor<=w['deadline_h']+1e-8
        assert abs(fine_carbon-fine_record['candidate']['carbon_g'])<1e-5
        best=min(s['carbon_g'],f['carbon_g'],fine_carbon); lb,res=lower_bound(menu,ci,w['tokens'],w['deadline_h'])
        fix=fixed(menu,ci,w['tokens'],w['deadline_h'])
        assert abs(fix[0]-s['best_fixed_carbon_g'])<1e-5
        r=dict(model=model,deadline_h=w['deadline_h'],best_fixed_h=fix[2],slack_h=w['deadline_h']-fix[2],slack_pct=100*(w['deadline_h']/fix[2]-1),fixed_carbon_g=fix[0],best_production_g=best,lp_lower_bound_g=lb,max_additional_saving_pct=max(0.,100*(best-lb)/best),max_total_saving_pct=max(0.,100*(fix[0]-lb)/fix[0]),production_saving_pct=100*(fix[0]-best)/fix[0],lp_token_residual=res,feasible_fixed_count=sum(w['tokens']/menu.throughput_tok_s/3600+menu.save_s/3600<=w['deadline_h']))
        for name,values in [('throughput_tok_s',menu.throughput_tok_s),('power_w',menu.power_w),('joules_per_token',menu.power_w/menu.throughput_tok_s)]:
            r[name+'_min']=float(values.min()); r[name+'_max']=float(values.max())
        metrics.append(r)
        cases=[('start_offset_h',v,w['deadline_h'],v) for v in [0,6,12,18]]
        cases += [('deadline_factor',v,math.ceil(v*w['fastest_static_h']),0) for v in [1.,1.25,1.5]]
        cases += [('decision_interval_h',v,w['deadline_h'],0) for v in [.25,1.,4.]]
        cache={(w['deadline_h'],0):(lb,res)}
        for factor,value,deadline,offset in cases:
            if (deadline,offset) not in cache: cache[deadline,offset]=lower_bound(menu,ci,w['tokens'],deadline,offset)
            bound,res=cache[deadline,offset]; feasible=fixed(menu,ci,w['tokens'],deadline,offset)
            pair=one_switch(menu,ci,w['tokens'],deadline,offset,value if factor=='decision_interval_h' else 1.)
            sensitivity.append(dict(model=model,factor=factor,value=value,one_switch_carbon_g=pair['carbon_g'],one_switch_first_action=pair['first_action'],one_switch_second_action=pair['second_action'],one_switch_at_h=pair['switch_h'],one_switch_completion_h=pair['completion_h'],one_switch_saving_pct=100*(feasible[0]-pair['carbon_g'])/feasible[0],deadline_h=deadline,offset_h=offset,lp_lower_bound_g=bound,feasible_fixed_carbon_g=feasible[0],feasible_fixed_action=feasible[1],feasible_fixed_completion_h=feasible[2],max_saving_vs_fixed_pct=100*(feasible[0]-bound)/feasible[0],lp_token_residual=res))
        print(model, r['max_additional_saving_pct'],flush=True)
    pd.DataFrame(metrics).to_csv(OUT/'opportunity_metrics.csv',index=False)
    pd.DataFrame(sensitivity).to_csv(OUT/'opportunity_sensitivity.csv',index=False)
if __name__=='__main__': run()

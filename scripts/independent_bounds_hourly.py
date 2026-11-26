"""Exact separable terminal-hour LP relaxation via convex hull resource allocation.
Independent of production code, no SciPy LP/MILP calls. Each complete hour is
convex mixture of all actions; terminal hour additionally allows zero duration.
"""
import json, math
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/measured_models_independent_audit'

def hull(points):
    # Lower convex rate/power frontier, retaining minimum-power equal-rate rows.
    unique={}
    for rate,power in points: unique[rate]=min(unique.get(rate,math.inf),power)
    out=[]
    for point in sorted(unique.items()):
        while len(out)>=2:
            a,b=out[-2:]
            if (b[0]-a[0])*(point[1]-b[1])-(b[1]-a[1])*(point[0]-b[0])>0: break
            out.pop()
        out.append(point)
    return np.array(out,dtype=float)

def terminal_bounds(w):
    p=pd.read_csv(w['profile']); trace=pd.read_csv(w['trace']); hours=int(w['deadline_h']); assert hours==w['deadline_h']
    assert (p[['save_s','warmup_wall_s','warmup_iters','reshard_wall_time_s']].to_numpy()==0).all()
    assert w['offset_h']==0 and w['wrap_trace']
    rates=p.throughput_tok_s.to_numpy()*3600/1e6; powers=p.power_w.to_numpy()/1000
    full=hull(zip(rates,powers)); final=hull([(0.,0.),*zip(rates,powers)])
    q=w['tokens']/1e6; ci=np.resize(trace.carbon.to_numpy(dtype=float),hours)
    full_width=np.diff(full[:,0]); full_slope=np.diff(full[:,1])/full_width
    last_width=np.diff(final[:,0]); last_slope=np.diff(final[:,1])/last_width
    first=max(0,math.ceil(q/max(rates))-1); results=[]
    for h in range(first,hours):
        baseline_work=h*full[0,0]; baseline_cost=sum(ci[:h])*full[0,1]
        remaining=q-baseline_work
        if remaining < -1e-8: continue
        slopes=np.concatenate([(ci[:h,None]*full_slope).ravel(),ci[h]*last_slope])
        capacities=np.concatenate([np.tile(full_width,h),last_width])
        order=np.argsort(slopes,kind='stable'); slopes=slopes[order]; capacities=capacities[order]
        cumulative=np.cumsum(capacities)
        if not len(cumulative) or remaining>cumulative[-1]+1e-8: continue
        k=min(int(np.searchsorted(cumulative,remaining)),len(cumulative)-1)
        used_before=cumulative[k-1] if k else 0.
        cost=baseline_cost+np.dot(slopes[:k],capacities[:k])+(remaining-used_before)*slopes[k]
        # Dual lambda certificate: lambda*target + sum min(cost-lambda*rate).
        lam=slopes[k]
        dual=lam*q+np.min(ci[:h,None]*powers[None,:]-lam*rates[None,:],axis=1).sum()+min(0.,np.min(ci[h]*powers-lam*rates))
        if abs(cost-dual)>1e-6: raise RuntimeError(f'Primal-dual mismatch {cost-dual}')
        results.append(dict(terminal_hour=h,lp_bound_g=float(dual),primal_g=float(cost),dual_lambda_g_per_million_tokens=float(lam),primal_dual_residual_g=float(cost-dual)))
    best=min(results,key=lambda x:x['lp_bound_g'])
    production=pd.read_csv(Path(w['out'])/'results.csv'); feasible=production[production.feasible]
    fixed=feasible[feasible.policy!='configuration_oracle'].carbon_g.min(); incumbent=feasible.carbon_g.min()
    independent=OUT/f"independent_solve_{w['model']}.json"
    if independent.exists(): incumbent=min(incumbent,json.loads(independent.read_text())['carbon_g'])
    lower=best['lp_bound_g']
    return dict(model=w['model'],target_tokens=w['tokens'],deadline_h=hours,action_count=len(rates),full_hour_hull=full.tolist(),terminal_hull=final.tolist(),lower_bound_g=lower,best_terminal_hour=best['terminal_hour'],incumbent_g=float(incumbent),best_fixed_g=float(fixed),achieved_saving_pct=float(100*(fixed-incumbent)/fixed),maximum_saving_pct=float(100*(fixed-lower)/fixed),unresolved_carbon_gap_g=float(incumbent-lower),terminal_bounds=results)

if __name__=='__main__':
    records=[]
    for path in sorted((ROOT/'runs/measured_models_corrected_west1').glob('*/workload.json')):
        result=terminal_bounds(json.loads(path.read_text())); records.append({k:v for k,v in result.items() if k not in ['terminal_bounds','full_hour_hull','terminal_hull']})
        (OUT/f"independent_bounds_{result['model']}.json").write_text(json.dumps(result,indent=2)+'\n')
        print(records[-1],flush=True)
    pd.DataFrame(records).to_csv(OUT/'independent_bounds_summary.csv',index=False)

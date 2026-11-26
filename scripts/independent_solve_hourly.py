"""Independent hourly MILP; no production simulator imports until replay.
Full-hour one-hot binaries, one terminal action, fractional terminal duration.
Token rows scaled by maximum hourly rate; objective is carbon grams.
"""
import argparse, json, math, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import milp, Bounds, LinearConstraint
from scipy.sparse import lil_matrix
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/measured_models_independent_audit'

def problem(rate,power,ci,target,terminal,interval_h):
    a=len(rate); h=terminal; n=(h+2)*a
    # x[hour,action], terminal selector z[action], duration fraction f[action].
    mat=lil_matrix((h+2+a,n)); lo=np.zeros(h+2+a); hi=np.zeros(h+2+a)
    for t in range(h):
        mat[t,t*a:(t+1)*a]=1; lo[t]=hi[t]=1
    mat[h,h*a:(h+1)*a]=1; lo[h]=hi[h]=1
    scale=max(rate)*3600*interval_h
    for t in range(h): mat[h+1,t*a:(t+1)*a]=rate*3600*interval_h/scale
    mat[h+1,(h+1)*a:]=rate*3600*interval_h/scale
    lo[h+1]=hi[h+1]=target/scale
    for j in range(a):
        mat[h+2+j,(h+1)*a+j]=1; mat[h+2+j,h*a+j]=-1
        lo[h+2+j]=-np.inf; hi[h+2+j]=0
    c=np.concatenate([np.concatenate([power*ci[t]*interval_h/1000 for t in range(h)]) if h else np.array([]),np.zeros(a),power*ci[h]*interval_h/1000])
    integrality=np.ones(n); integrality[(h+1)*a:]=0
    return c, integrality, Bounds(np.zeros(n),np.ones(n)),LinearConstraint(mat.tocsc(),lo,hi)

def solve(w, interval_h=1, time_limit=30):
    profile=pd.read_csv(w['profile']); rate=profile.throughput_tok_s.to_numpy(); power=profile.power_w.to_numpy()
    assert (profile[['save_s','warmup_wall_s','warmup_iters','reshard_wall_time_s']].to_numpy()==0).all()
    trace=pd.read_csv(w['trace']); ci0=trace.carbon.to_numpy()
    # This public implementation supports hourly intervals only, no trace interpolation.
    assert interval_h==1
    horizon=int(w['deadline_h']); ci=np.resize(ci0,horizon); a=len(rate); target=w['tokens']
    rows=[]; incumbent=None; best=math.inf
    # A feasible independent static candidate supplies an initial upper bound.
    for j in range(a):
        duration=target/rate[j]/3600
        if duration<=horizon:
            k=int(duration); frac=duration-k
            cost=power[j]/1000*(sum(ci[:k])+(ci[k]*frac if frac else 0))
            if cost<best: best=cost; incumbent=dict(actions=[j]*(k+(frac>0)),fractions=[1.]*k+([frac] if frac else []),carbon_g=cost)
    for h in range(max(0,math.ceil(target/max(rate)/3600)-1),horizon):
        c,integ,bounds,cons=problem(rate,power,ci,target,h,interval_h)
        lp=milp(c,integrality=np.zeros(len(c)),bounds=bounds,constraints=cons)
        if lp.status not in (0,2): raise RuntimeError(f'Uncertified LP status {lp.status}: {lp.message}')
        rows.append(dict(terminal_hour=h,lp_status=int(lp.status),lp_bound=float(lp.fun) if lp.fun is not None else None))
    for row in sorted(rows,key=lambda r:r['lp_bound'] if r['lp_bound'] is not None else math.inf):
        h=row['terminal_hour']; lower=row['lp_bound']
        if lower is None or lower>=best-1e-7: row['status']='LP pruned'; continue
        c,integ,bounds,cons=problem(rate,power,ci,target,h,interval_h)
        res=milp(c,integrality=integ,bounds=bounds,constraints=cons,options={'time_limit':time_limit,'mip_rel_gap':1e-9})
        dual=getattr(res,'mip_dual_bound',None)
        dual=float(dual) if dual is not None and np.isfinite(dual) else lower
        row.update(status=int(res.status),message=res.message,mip_bound=max(lower,dual),mip_objective=float(res.fun) if res.fun is not None else None,mip_gap=float(res.mip_gap) if getattr(res,'mip_gap',None) is not None else None)
        if res.x is not None:
            residual=cons.A@res.x
            row['max_integrality_error']=float(np.max(np.abs(res.x[:(h+1)*a]-np.rint(res.x[:(h+1)*a]))))
            row['max_constraint_violation']=float(max(np.max(cons.lb-residual),np.max(residual-cons.ub),0))
            if row['max_integrality_error']>1e-5 or row['max_constraint_violation']>1e-5: raise RuntimeError('Invalid incumbent')
        print(w['model'],h,row,flush=True)
        if res.x is not None and res.fun<best:
            actions=list(np.argmax(res.x[:h*a].reshape(h,a),axis=1).astype(int)); last=int(np.argmax(res.x[h*a:(h+1)*a])); actions.append(last)
            fractions=[1.]*h+[float(res.x[(h+1)*a+last])]
            best=float(res.fun); incumbent=dict(actions=[int(j) for j in actions],fractions=fractions,carbon_g=best)
    global_lower=min(r.get('mip_bound',r['lp_bound']) for r in rows if r['lp_bound'] is not None)
    ledger=[]; tokens=energy=carbon=elapsed=0.
    for h,(j,f) in enumerate(zip(incumbent['actions'],incumbent['fractions'])):
        # Independently reconstruct the last duration from the exact remaining tokens.
        duration=min(3600.,(target-tokens)/rate[j]); dtokens=rate[j]*duration; de=power[j]*duration; dc=de*ci[h]/3.6e6
        if duration < -1e-6: raise RuntimeError('Negative duration')
        tokens+=dtokens; energy+=de; carbon+=dc; elapsed+=duration
        ledger.append(dict(hour=h,action=j,duration_s=duration,rate_tok_s=rate[j],power_w=power[j],ci_g_kwh=ci[h],interval_tokens=dtokens,tokens_completed=tokens,interval_energy_j=de,energy_j=energy,interval_carbon_g=dc,carbon_g=carbon))
    if abs(tokens-target)>1e-5 or elapsed>horizon*3600+1e-6: raise RuntimeError('Invalid reconstructed schedule')
    result=dict(model=w['model'],target_tokens=target,deadline_h=horizon,action_count=a,carbon_g=carbon,completion_h=elapsed/3600,lower_bound_g=global_lower,signed_bound_difference_g=carbon-global_lower,absolute_gap_g=max(0.,carbon-global_lower),relative_gap=max(0.,carbon-global_lower)/carbon,terminal_bounds=rows,actions=incumbent['actions'],ledger=ledger)
    return result

def replay(w,result):
    sys.path.insert(0,str(ROOT))
    from evaluate_carbon_policy import load_strategy_profile
    from carbon_aware_3d.carbon_aware_env import CarbonSim,SimConfig,load_carbon_trace
    cfg=SimConfig(load_strategy_profile(w['profile']),w['tokens'],w['global_batch_size']*w['seq_length'])
    t,ci,_=load_carbon_trace(w['trace']); sim=CarbonSim(cfg,t,ci,wrap_trace=True); sim.reset(0,result['actions'][0])
    for j in result['actions']:
        if not sim.done(): sim.step(j)
    discrepancies=[]
    for actual,expected in zip(sim.history,result['ledger']):
        for field in ['tokens_completed','energy_j','carbon_g']:
            delta=getattr(actual,field)-expected[field]
            if abs(delta)>max(1e-6,abs(expected[field])*1e-12): discrepancies.append(dict(hour=actual.step,field=field,delta=delta))
    return dict(carbon_g=sim.total_carbon_g,completion_h=sim.cursor_s/3600,tokens_completed=sim.tokens_completed,completed=sim.done(),history_length=len(sim.history),discrepancies=discrepancies)

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--model',default='gpt2'); p.add_argument('--time-limit',type=float,default=30); args=p.parse_args()
    w=json.loads((ROOT/f'runs/measured_models_corrected_west1/{args.model}/workload.json').read_text())
    result=solve(w,time_limit=args.time_limit); result['replay']=replay(w,result)
    OUT.mkdir(exist_ok=True); pd.DataFrame(result.pop('ledger')).to_csv(OUT/f'independent_solve_{args.model}_ledger.csv',index=False)
    (OUT/f'independent_solve_{args.model}.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['terminal_bounds','actions']},indent=2))

"""Independent predictions and production objective checks; no existing outputs modified."""
import json, sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from evaluate_carbon_policy import load_strategy_profile
from carbon_aware_3d.carbon_aware_env import SimConfig, rollout_oracle_zero_overhead, load_carbon_trace, rollout_static
OUT=ROOT/'runs/measured_models_independent_audit'
def solve(cfg, ci, hours, q):
    return rollout_oracle_zero_overhead(cfg,np.arange(len(ci))*3600.,np.asarray(ci,dtype=float),0,max_hours=hours,token_quantum=q,wrap_trace=True)
def result(s):
    return dict(carbon_g=s.total_carbon_g,completion_h=s.cursor_s/3600,actions=[r.strategy_idx for r in s.history])
def strategy(rate,power):
    return dict(tp=1,pp=1,dp=8,throughput_tok_s=rate/3600,power_w=power,warmup_wall_s=0,warmup_iters=0,save_s=0)
# Dirty first, clean second: slow first uses slack rather than racing to finish.
# A=2 tokens/h, 2000W; B=1 token/h,500W; target 3, deadline2.
# AA=20+1=21g, AB=20+.5=20.5g, BA=5+2=7g; BB infeasible.
cfg=SimConfig([strategy(2,2000),strategy(1,500)],3,1)
s=solve(cfg,[10,1],2,1)
assert [r.strategy_idx for r in s.history]==[1,0]
assert abs(s.total_carbon_g-7)<1e-10 and s.cursor_s==7200
report={'hand_calculated':dict(predicted_g=7,fast_fixed_g=21,slack_used_h=.5,**result(s))}
coarse=solve(cfg,[10,1],2,1.1)
assert abs(coarse.total_carbon_g-21)<1e-10
report['rounding_reproducer']=dict(quantum=1.1,exact_optimum_g=7,excess_carbon_g=14,**result(coarse))
static_cfg=SimConfig([strategy(1,1000)],10,1)
overrun=rollout_static(static_cfg,np.arange(4)*3600.,np.ones(4),0,0,max_hours=1)
assert overrun.cursor_s==10800 and overrun.tokens_completed==3
report['static_horizon_reproducer']=dict(requested_max_hours=1,actual_hours=3,tokens=overrun.tokens_completed,carbon_g=overrun.total_carbon_g,deadline_clipped_carbon_g=1,main_runner_winner_impact_g=0)
for model in ['gpt2','llama_1b']:
    d=ROOT/'runs/measured_models_corrected_west1'/model
    w=json.loads((d/'workload.json').read_text())
    cfg=SimConfig(load_strategy_profile(d/'strategy_profile.csv'),w['tokens'],w['global_batch_size']*w['seq_length'])
    _,ci,_=load_carbon_trace(w['trace'])
    base=solve(cfg,ci,w['deadline_h'],w['token_quantum'])
    scales=[]
    for scale in [.1,7.3]:
        scaled=solve(cfg,ci*scale,w['deadline_h'],w['token_quantum'])
        assert np.isclose(scaled.total_carbon_g,scale*base.total_carbon_g,rtol=1e-11)
        assert result(scaled)['actions']==result(base)['actions']
        scales.append(dict(scale=scale,**result(scaled)))
    efficient=min(cfg.strategies,key=lambda a:a['power_w']/a['throughput_tok_s'])
    assert w['tokens']/efficient['throughput_tok_s']<=w['deadline_h']*3600
    prediction=w['tokens']*efficient['power_w']/efficient['throughput_tok_s']*100/3.6e6
    flat=solve(cfg,np.full(744,100.),w['deadline_h'],w['token_quantum'])
    assert np.isclose(prediction,flat.total_carbon_g,rtol=1e-11)
    report[model]=dict(base=result(base),scaled=scales,flat=result(flat),flat_predicted_g=prediction)
(OUT/'objective_checks.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k: {a:b for a,b in v.items() if a not in ['scaled','base','flat']} for k,v in report.items()},indent=2))

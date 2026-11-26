"""Independent direct GPT-2 one-switch enumeration (not an optimizer search)."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from evaluate_carbon_policy import load_strategy_profile
from carbon_aware_3d.carbon_aware_env import CarbonSim, SimConfig, load_carbon_trace, rollout_oracle_zero_overhead

KEYS=("tp","pp","dp","micro_batch_size","placement_id")
def replay(cfg,t,ci,a,b,boundary,deadline):
    sim=CarbonSim(cfg,t,ci,wrap_trace=True)
    sim.reset(0,initial_strategy_idx=a)
    for hour in range(deadline):
        sim.step(a if hour<boundary else b,3600)
        if sim.tokens_completed>=cfg.pretraining_tokens-1e-6: break
    return dict(first=a,second=b,boundary_h=boundary,carbon_g=sim.total_carbon_g,
                completion_h=sim.cursor_s/3600,tokens_done=sim.tokens_completed,
                feasible=bool(sim.tokens_completed>=cfg.pretraining_tokens-1e-6 and sim.cursor_s<=deadline*3600+1e-6))

def audit():
    profile=ROOT/"runs/measured_models_fast/gpt2/strategy_profile.csv"
    strategies=load_strategy_profile(profile)
    assert len(strategies)==38
    cfg=SimConfig(strategies,2489518080,16*1024)
    assert all(s.get(k,0)==0 for s in strategies for k in ("warmup_wall_s","warmup_iters","reshard_wall_time_s","save_s"))
    report={"tokens":cfg.pretraining_tokens,"deadline_h":9,"action_count":38,"traces":{}}
    for name,path in [("original","data/carbon_2024_05_hourly_aws_us-west-1.csv"),("recovered","data/recovered_workshop_may2024/carbon_2024_05_hourly_aws_us-west-1.csv")]:
        t,ci,_=load_carbon_trace(ROOT/path)
        static=[replay(cfg,t,ci,a,a,9,9) for a in range(38)]
        pairs=[replay(cfg,t,ci,a,b,h,9) for a in range(38) for b in range(38) if a!=b for h in range(1,9)]
        feasible_static=[r for r in static if r["feasible"]]
        feasible_pairs=[r for r in pairs if r["feasible"]]
        best_static=min(feasible_static,key=lambda r:r["carbon_g"])
        best_pair=min(feasible_pairs,key=lambda r:r["carbon_g"])
        actual=[r for r in feasible_pairs if r["completion_h"]>r["boundary_h"]]
        best_actual=min(actual,key=lambda r:r["carbon_g"])
        for r in [best_static,best_pair,best_actual]:
            r["first_tuple"]={k:strategies[r["first"]][k] for k in KEYS}
            r["second_tuple"]={k:strategies[r["second"]][k] for k in KEYS}
        report["traces"][name]={"path":path,"sha256":hashlib.sha256((ROOT/path).read_bytes()).hexdigest(),
           "static_count":38,"feasible_static":len(feasible_static),"infeasible_static":38-len(feasible_static),
           "pair_count":len(pairs),"feasible_pairs":len(feasible_pairs),"infeasible_pairs":len(pairs)-len(feasible_pairs),
           "feasible_actual_switches":len(actual),"best_static":best_static,"best_pair":best_pair,"best_actual_switch":best_actual,
           "pairs_beating_static":sum(r["carbon_g"]<best_static["carbon_g"]-1e-8 for r in feasible_pairs)}
    # A produces 2 tokens/hour at 2 kW; B produces 1 at 0.5 kW.
    # Three tokens, two-hour deadline, CI [1,10]: A->B costs 2+5=7g.
    # Static A costs 2+10=12g; B unfinished; B->A costs 0.5+20=20.5g.
    tiny=[dict(tp=1,pp=1,dp=1,power_w=p,throughput_tok_s=q/3600,warmup_wall_s=0,warmup_iters=0,save_s=0,reshard_wall_time_s=0) for p,q in [(2000,2),(500,1)]]
    toy=SimConfig(tiny,3,1,n_gpus=1)
    t=np.array([0.,3600.]); ci=np.array([1.,10.])
    oracle=rollout_oracle_zero_overhead(toy,t,ci,0,max_hours=2,token_quantum=1)
    direct=replay(toy,t,ci,0,1,1,2)
    assert abs(direct["carbon_g"]-7)<1e-10
    assert abs(oracle.total_carbon_g-7)<1e-10
    assert [r.strategy_idx for r in oracle.history]==[0,1]
    report["tiny_known_winner"]={"expected_carbon_g":7,"static_A_carbon_g":12,"B_to_A_carbon_g":20.5,"direct":direct,"production_carbon_g":oracle.total_carbon_g,"production_actions":[r.strategy_idx for r in oracle.history]}
    out=ROOT/"runs/measured_models_audit/gpt2_pairs.json"
    out.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))
if __name__=="__main__": audit()

import csv, json, hashlib, math, sys
from pathlib import Path
root=Path(sys.argv[1]); run=root / sys.argv[2]
model=run/'llama_3b'
w=json.loads((model/'workload.json').read_text())
trace=Path(w['trace']); ci=[float(r['carbon']) for r in csv.DictReader(trace.open())]
profiles=list(csv.DictReader((model/'strategy_profile.csv').open()))
rows=list(csv.DictReader((model/'schedules.csv').open()))
summary=json.loads((model/'summary.json').read_text())
policies={r['policy'] for r in rows if r['policy']=='configuration_oracle' or (r['strategy_idx']==str(summary['best_fixed_idx']) and r['policy'].startswith('static'))}
out={'trace':str(trace),'sha256':hashlib.sha256(trace.read_bytes()).hexdigest(),'rows':len(ci),'mean_ci':sum(ci)/len(ci),'min_ci':min(ci),'max_ci':max(ci),'policies':{}}
for policy in sorted(policies):
    energy=carbon=0.; max_ci_error=max_energy_error=max_carbon_error=0.
    selected=[r for r in rows if r['policy']==policy]
    for r in selected:
        a,b=float(r['t_start_s']),float(r['t_end_s']); power=float(profiles[int(r['strategy_idx'])]['power_w'])
        e=power*(b-a); c=0.; t=a
        while t < b:
            hour=math.floor(t/3600); end=min(b,(hour+1)*3600)
            c+=power*(end-t)*ci[hour%len(ci)]/3600000; t=end
        energy+=e; carbon+=c
        max_ci_error=max(max_ci_error,abs(float(r['ci_at_step_start'])-ci[math.floor(a/3600)%len(ci)]))
        max_energy_error=max(max_energy_error,abs(e-float(r['interval_energy_j'])))
        max_carbon_error=max(max_carbon_error,abs(c-float(r['interval_carbon_g'])))
    end=float(selected[-1]['t_end_s']); tail=end/3600%len(ci); n=math.floor(tail); fraction=tail-n
    tail_sum=sum(ci[:n])+fraction*ci[n]
    out['policies'][policy]={'rows':len(selected),'energy_j':energy,'carbon_g':carbon,'saved_carbon_g':float(selected[-1]['carbon_g']),'carbon_delta_g':carbon-float(selected[-1]['carbon_g']),'effective_ci':carbon/(energy/3600000),'completion_h':end/3600,'full_wraps':int(end/3600//len(ci)),'tail_hours':tail,'tail_mean_ci':tail_sum/tail,'tail_final_hour_ci':ci[n],'max_step_ci_error':max_ci_error,'max_interval_energy_error_j':max_energy_error,'max_interval_carbon_error_g':max_carbon_error}
print(json.dumps(out,indent=2))

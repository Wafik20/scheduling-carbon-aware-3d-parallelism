# Audit of all-zero savings in measured_models_fast

## Confirmed discrepancy and fix

**Confirmed bug:** wrong default trace selection. **Not found:** a unit/PUE/wrap/label-mapping error or automatic prohibition on switching. **Numerical limitation confirmed:** default token rounding can miss a beneficial switched schedule; zero savings is not an optimality certificate.

The runner default selected the suspect top-level west-1 trace, not the recovered workshop trace. The three top-level regional May files have the same SHA-256 `4d38312539f016134419af012d48ec2f3cd51227fd1793412ef67f075289cda9`, mean 385.241935483871 gCO2/kWh, range 306–468. Recovered workshop west-1 has SHA-256 `8fb0da0d267112b666d726298746f4391804ca619b3924a9a1dec279666db96e`, mean 142.728494623656, range 55–315. Its Git blob matches historical commit `6c27ee4`. This verifies historical recovery, not external grid-zone provenance; see trace_provenance.md.

Fixed the default to the recovered file and added per-workload trace hash/mean plus console reporting. Old output directory remains unchanged; rerun: `runs/measured_models_corrected_west1`.

## Independent accounting

Search and replay consume the same hourly arrays. No PUE multiplier or average-CI substitution is applied. Carbon is sum(power_W * seconds * CI_g_per_kWh / 3,600,000). Power is measured GPU power, not facility power.

Original Llama-3B: 7,233,011,272.297247 J = 2009.169797860346 kWh, duration 1657.047255967296 hours, action 31 at 1212.5 W. Independently integrated carbon 777677.1609842276 g versus saved 777677.1609842277 g. Effective CI is 387.063931486732. The run consumes 1488 hours (two cycles) plus 169.047255967296 hours; tail mean 403.101636664163 explains the difference from the monthly mean. Last partial hour: 0.047255967296 hour at CI 428. On recovered west-1 the same fixed schedule gives 289063.23824324214 g and effective CI 143.871980631542. Wrapping and the partial final hour are correct.

## Original trace: retained search candidate vs fixed

Every retained rounded-DP winner is itself static. Therefore no switched winner reached replay and then vanished in the static fallback. The search does not retain a separate best-switched runner-up. Equal rounded-cost states retain the first enumerated action; only one best terminal path is replayed. Other paths are pruned at the same (hour, token-bin) state. Equal exact carbon favors the fixed baseline. Code locations: `carbon_aware_env.py:929–952` enumerates actions, floors progress and retains strict improvements; `:976–982` replays and compares against static. The runner loads one trace at `scripts/evaluate_measured_models.py:243` and passes it to the oracle at `:273`. This is not proof that every arbitrary continuous-progress switching schedule loses.

|Model|Best fixed / exact DP replay g|Rounded DP cost g|Replay switches|Decision|
|---|---:|---:|---:|---|
|gpt2|2109.231172186|2128.821613694|0|Equal; fixed retained|
|gpt2_large|40507.983907546|47645.706740368|0|Equal; fixed retained|
|gpt2_medium|11223.670779541|11250.593579877|0|Equal; fixed retained|
|llama_1b|154562.925502336|163525.343265153|0|Equal; fixed retained|
|llama_2b|457285.493346368|468017.752522807|0|Equal; fixed retained|
|llama_3b|777677.160984228|806650.431152296|0|Equal; fixed retained|
|pythia_1_4b|91503.680662139|92755.624472873|0|Equal; fixed retained|
|pythia_2_8b|318797.688063955|338939.338278689|0|Equal; fixed retained|
|pythia_410m|13247.216909099|13564.322515230|0|Equal; fixed retained|

## Recovered west-1: candidate replay and selected results

|Model|Best fixed g|DP candidate replay g|Candidate switches|Selected savings %|Decision|
|---|---:|---:|---:|---:|---|
|gpt2|968.542786568|968.542786568|0|0.000000|Equal|
|gpt2_large|16299.910953773|16971.309855649|8|0.000000|Candidate higher; fixed retained|
|gpt2_medium|4675.115499662|4675.115499662|0|0.000000|Equal|
|llama_1b|64020.830408352|63729.071428306|10|0.455725|Candidate lower|
|llama_2b|177336.316111723|175907.075118673|25|0.805949|Candidate lower|
|llama_3b|289063.238243242|289111.599861543|12|0.000000|Candidate higher; fixed retained|
|pythia_1_4b|35047.250673734|35168.003508800|8|0.000000|Candidate higher; fixed retained|
|pythia_2_8b|119699.661433739|117082.800125200|31|2.186189|Candidate lower|
|pythia_410m|5316.375487243|5316.375487243|0|0.000000|Equal|

All candidates in these tables complete within the common deadline. No label-based action collapse occurs: integer source-row indices carry all TP/PP/DP/MBS/placement variants through DP, parents and replay. Equivalent rank-order aliases remain metadata. Whole-run static infeasibility does not exclude an action from a single hour. All switching/startup/build/cache/warmup/save durations are zero for this experiment (measured save_s is zero in all 348 rows); idle power is configured but never consumes time. The horizon and last partial decision enforce the deadline.

## Independent GPT-2 exhaustive one-switch check

For the exact 2,489,518,080-token, 9-hour workload, all 38*37*8 = 11,248 ordered distinct-action pairs and boundaries 1..8 hours were replayed directly. Per trace: 2,680 feasible paths; 2,458 actually reach a switch; 222 finish before their switch. No path beats fixed action22. Old trace fixed = 2109.2311721855867 g; best actual switch22→23 at6h = 2126.505032098841 g. Recovered trace fixed = 968.5427865678323 g; best actual switch = 979.4859124814647 g. These actions share TP1/PP4/DP2/MBS2 but differ in placement: tp-dp-pp vs tp-pp-dp. This verifies a genuine fixed winner within the exhaustive one-switch class, not arbitrary multi-switch optimality. Of the feasible actual-switch schedules, 1,468 use at least one action infeasible as a whole-run fixed policy. Example: action4 alone produces only 2,294,796,096 tokens by9h, but action4→22 at1h finishes in6.97171643548h (2173.93445912608g on the original trace). Thus whole-run static infeasibility does not prevent hourly use.

Synthetic test: A=2 tokens/hour, 2000W; B=1 token/hour,500W; target3 tokens, deadline2h, CI=[1,10]. A→B costs2+5=7g; fixed A costs2+10=12g; B→A costs0.5+20=20.5g; fixed B infeasible. Production search returns[0,1],7g.

## Token-quantum sensitivity

Tenfold finer bins retain zero savings for all nine models on the original suspect trace. On recovered west-1, Llama-3B changes from zero to a small positive saving, demonstrating that its default-Q zero was an approximation artifact, not proof of a genuine global tie. Other exact replay scores are not monotone in Q: the algorithm retains only the rounded-objective winner, and floor accumulation changes rankings. No change to the discretization algorithm was made in this audit. The default and finer results should both be reported, not silently cherry-picked.

|Model|Default selected savings %|Q/10 selected savings %|Q/10 candidate replay g|Q/10 switches|
|---|---:|---:|---:|---:|
|gpt2|0.000000|0.000000|968.542786568|0|
|gpt2_large|0.000000|0.000000|16299.910953773|0|
|gpt2_medium|0.000000|0.000000|4675.115499662|0|
|llama_1b|0.455725|0.332885|63807.714489852|16|
|llama_2b|0.805949|0.791817|175932.137799699|24|
|llama_3b|0.000000|0.026781|288985.823941962|16|
|pythia_1_4b|0.000000|0.000000|35047.250673734|0|
|pythia_2_8b|2.186189|2.251291|117004.873168047|42|
|pythia_410m|0.000000|0.000000|5316.375487243|0|

Llama-3B Q=200,000 replay: 288985.8239419617g vs fixed289063.23824324214g, 16 switches, saving77.414301280456g (0.026781%). Saved production fine rerun: `runs/measured_models_corrected_west1_fine`. Its action menu, deadline, workload and trace are unchanged.

## Physical tradeoffs and historical comparison

|Model|Fastest tok/s|Fastest W|Fastest J/token|Most efficient action J/token|Power/rate frontier actions|
|---|---:|---:|---:|---:|---:|
|gpt2|103940.93|763.8|0.007348405|0.007348405|13|
|gpt2_large|43693.31|1000.4|0.022895954|0.022895954|20|
|gpt2_medium|62657.62|863.1|0.013774861|0.013774861|19|
|llama_1b|20932.88|1024.4|0.048937366|0.047818572|17|
|llama_2b|12788.14|1055.1|0.082506135|0.082506135|15|
|llama_3b|12092.30|1212.5|0.100270420|0.100270420|13|
|pythia_1_4b|45078.12|1301.5|0.028872100|0.028872100|14|
|pythia_2_8b|26273.65|1559.1|0.059340822|0.053174004|9|
|pythia_410m|78187.85|1113.3|0.014238785|0.014238785|22|

Detailed per-model rate/power/J-token values and frontiers: `runs/measured_models_audit/physical_tradeoffs.csv` and `history_tradeoffs.md`. Tradeoff calculations use power_w/throughput_tok_s, the exact simulator quantities; recorded joules_per_token is rounded separately and is not an extra energy multiplier. Seven models have the same fastest and most energy-efficient action. Llama-1B trades 1.60647% rate for 2.28617% lower J/token; Pythia-2.8B trades 5.17332% rate for 10.39220% lower J/token. Every model has several power/rate-frontier actions. Dirty-hour lower power plus clean-hour catch-up can help even when the slow action is less energy-efficient; none of these comparisons proves zero savings for varying CI.

Historical 2.5877160383614175%: `6c27ee4:figures/experiment_plots/regions/faithful_token_sweep/faithful_token_sweep_feasible_summary.csv`, GPT-2 west-1 8B tokens,26h,64 random episodes seed0 over shared offset cap306.5h. The statistic is conditional on50/64 feasible dynamic episodes;26 of those50 never switched. Paired static mean6160.824743429445g, dynamic5995.320034078640g; the quoted2.59% is mean per-episode percentage, not ratio of means. Historical quantum25M, measured profile warmup/switch/save enabled, no wrapping; current workload is2.4895B/9h,offset0,38 measured actions,Q5M,zero overhead and new algorithm. Historical profile `runs/fixed_dp_25slack_gpt2_3p5b/strategy_profile.csv` is missing both locally and from Git, so its exact menu and costs cannot be established. Do not claim an exact reproduction from the available aggregates.

## Reproduction

```bash
.venv/bin/python scripts/audit_measured_carbon.py "$PWD" runs/measured_models_fast
.venv/bin/python scripts/audit_gpt2_switch_pairs.py
.venv/bin/python scripts/audit_oracle_candidates.py --factor 1 --trace old
.venv/bin/python scripts/audit_oracle_candidates.py --factor 10 --trace recovered
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -u scripts/evaluate_measured_models.py --workers 16 --out runs/measured_models_corrected_west1
```

## Validation and preservation

The full CPU suite passed 381 tests after the default/provenance fix. The new
trace regression verifies the recovered blob hash, 744 samples and expected
mean. Original `runs/measured_models_fast` contents (74 files) were hashed before
and after this audit and are unchanged; hashes are saved in
`runs/measured_models_audit/original_output_hashes.json`.

The region fix changes only the runner default and exposes trace hash/mean; it
does not overwrite any trace CSV or alter the production oracle. Fine-resolution
production rerun command:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -u scripts/evaluate_measured_models.py --models llama_3b --oracle-token-quantum 200000 --workers 1 --out runs/measured_models_corrected_west1_fine
```

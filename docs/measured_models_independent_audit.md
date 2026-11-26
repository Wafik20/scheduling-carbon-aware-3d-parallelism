# Independent audit of measured carbon-aware training

## Bottom line

Small savings mostly hold for this **GPU-only, stationary-profile, fixed-start, zero-switch-cost counterfactual**, but the production token-bin search misses a better Llama-1B schedule. Four of the five previously zero-saving models now have numerical optimality certificates for arbitrary hourly switches. Pythia-1.4B remains unresolved, with at most 0.033782% total possible savings. These are optimization claims conditional on measured inputs, not statistical or physical carbon-validation claims.

New artifacts only: `runs/measured_models_independent_audit/`. Existing fast, corrected and fine outputs were preserved. Production code and scientific constants were not changed. SciPy was installed in the project `.venv` for independent LP/MILP checks.

## 1. Raw reconstruction and measurement limitations

GPT2 case `gpt2_tp1_pp1_dp8_mbs1_tp-dp-pp`, attempt0001:

* Global batch is16, not16 per replica: MBS1 × DP8 × accumulation2. Sequence1024 gives16,384 global tokens/optimizer iteration. No additional DP multiplier.
* 225 measured iterations give3,686,400 tokens. Logged61,205.97tokens/s implies60.229418797s, consistent with rounded60.23s measurement time.
* Logged/calibration total power706.7W is the sum over eight distinct GPU UUIDs. Independently computed energy/token=706.7/61205.97=0.011546259294641J/token. Logged0.011547 is independently rounded.
* One hour gives220,341,492tokens,2,544,120J=0.7067kWh, and73.4968g at first-hour CI104g/kWh. An explicit production `CarbonSim.step` agrees exactly.

All348 tuples pass global-batch/DP, calibration-to-CSV, rank/GPU uniqueness and placement-group checks. No duplicated tuple, equivalent measured placement collision, or identical measurement vector was found. Across696 independent sidecar CSVs and313,409 samples, node totals equal their four GPU columns; all eight measured UUIDs are covered.

**Unavailable raw evidence matters:** the exact in-process power samples used for calibration were not saved. Sidecars include startup/warmup/cleanup and are separate samplers; their whole-file mean cannot reconstruct steady-phase power. Exact internal phase timestamps are absent. Measurement windows are60–62.71s, with one observation per tuple.33/348 warmups did not converge. No defensible empirical confidence interval follows from these inputs. The2% rank0 stationarity criterion is not an error bar. Some Pythia-2.8B switched schedules use a nonconverged action. Small gains cannot be claimed to exceed measurement uncertainty.

Recovered trace bytes match historical Git commit6c27ee4 and the known SHA256;744 contiguous relative hourly timestamps are correctly loaded. This proves recovery, **not** external grid-zone provenance. Timezone-free CSVs, absent source API responses and absent verified AWS-to-grid mapping prevent absolute timezone/region validation. May2024 repeated cyclically against October2026 measured hardware is a counterfactual, not contemporaneous measured carbon. GPU energy excludes hosts, network, cooling and PUE.

Full evidence, source hashes, all-row checks and limitations: [measurement report](../runs/measured_models_independent_audit/measurement_report.md).

## 2. Independent exact-progress optimization

Independent MILPs enumerate terminal hours. Each full hour chooses one action; the final hour chooses one action with a continuous duration. Token equality uses unrounded throughput. All configurations remain available, with no idle or delayed starts. LP/MIP dual bounds cover every terminal hour. Certificates use floating-point solver tolerances, not rational arithmetic.

**GPT2:**2,489,518,080tokens,9h deadline,38actions. Optimal schedule is action22 throughout (TP1/PP4/DP2/MBS2, placement `tp-dp-pp`), finishing6.653132698h. Carbon968.542786568g; certified gap1.14e-13g. Later terminal-hour relaxations cost at least1014.602646g or1157.369475g. There is no missed multi-switch saving.

Hourly ledger (power763.8W and rate103,940.93tokens/s throughout):

| Hour | Action | Duration h | Tokens | CI g/kWh | Emissions g |
|---|---:|---:|---:|---:|---:|
|0|22|1|374187348|104|79.435200|
|1|22|1|374187348|149|113.806200|
|2|22|1|374187348|206|157.342800|
|3|22|1|374187348|213|162.689400|
|4|22|1|374187348|214|163.453200|
|5|22|1|374187348|224|171.091200|
|6|22|0.653132698|244393992|242|120.724787|

**Llama-1B actual positive case:**29,969,653,760tokens,498h,41actions. Independent optimum63648.118077182g, finish433.980176933h, actions25/26/27, numerical optimality gap zero. Production's63729.071428306g misses80.953351124g. Fixed-relative saving rises from0.455725% to**0.582174%** (+0.126448 percentage points).

Both independent schedules replay in production with no hourly token, energy or carbon disagreement. The first **search** difference is Llama hour36: exact action25 versus production26 atCI262. Q=7M floors action25's19.856952M hourly tokens to14M. Action27's11.150676M floors to7M, becoming artificially dominated on the lattice. Thus search, not simulator accounting, loses the schedule. Complete434-row ledger, action sequence and terminal certificates are saved.

[Independent solve and reproduction commands](../runs/measured_models_independent_audit/independent_report.md).

## 3. Objective predictions and minimal reproducers

Flat100g/kWh gives exactly tokens × minimum feasible J/token ×100/3.6e6: GPT2=508.166275441g; Llama1B=39808.501279370g. Production matches. Scaling real CI by0.1 or7.3 preserves both selected schedules and scales emissions proportionally.

Hand example: A=2tokens/h at2000W, B=1token/h at500W, target3, deadline2h, CI=[10,1]. FixedA costs21g and finishes1.5h; B→A costs7g and finishes2h. Production returns B→A, explicitly using0.5h slack to move work into the clean hour.

Confirmed limitations/defects and quantified reproductions:

1. **Token rounding:** on the hand example, changing Q from1 to1.1 makes production return21g instead of7g (+14g). Real Llama miss above quantifies its actual impact. No claim that this synthetic200% excess applies to measured models.
2. **Static helper horizon:** `rollout_static(max_hours=1)` can run3h because it adds two full safety steps. A1token/h,1000W,target10,CI1 case returns3tokens/3g rather than horizon-clipped1token/1g. Selected baseline impact here is zero: the runner separately rejects completions after the deadline. Infeasible rows can overrun.
3. **Previously identified wrong trace selection:** regional files are identical; recovery provenance confirmed. Previous corrected-versus-original results quantify the impact. No new trace replacement was made.

Run `.venv/bin/python scripts/audit_independent_objective.py`. Details: [objective report](../runs/measured_models_independent_audit/objective_report.md). Relevant existing tests:54 passed. All six new analysis scripts compile successfully. SHA256 verification found no changes in182 existing fast/corrected/fine output files; see `preservation_check.json`.

## 4. Tight opportunity bounds

Relax each full hour to a convex mixture of measured actions, but keep it fully occupied; permit partial activity only in the terminal hour. Enumerate every possible terminal hour. The per-hour rate/power lower convex envelope gives an LP lower bound, independently checked by its Lagrange dual. This contains every legal hourly schedule without introducing arbitrary idle hours.1036 terminal bounds pass primal/dual checks; GPT2 and Llama bounds also agree with independent HiGHS LPs.

The following uses the best verified feasible incumbent (coarse/fine production, plus exact Llama1B), and the exact Llama bound where available. Savings use best fixed as denominator; remaining grams are the most any further hourly search could save.

|Model|Achieved savings %|Total savings ceiling %|Remaining carbon ceiling g|
|---|---:|---:|---:|
| gpt2 | 0.000000 | 0.000000 | 0.000000 |
| gpt2_large | 0.000000 | 0.000000 | 0.000000 |
| gpt2_medium | 0.000000 | 0.000000 | 0.000000 |
| llama_1b | 0.582174 | 0.582174 | 0.000000 |
| llama_2b | 0.805949 | 0.891879 | 152.383761 |
| llama_3b | 0.026781 | 0.139877 | 326.918579 |
| pythia_1_4b | 0.000000 | 0.033782 | 11.839747 |
| pythia_2_8b | 2.251291 | 2.623301 | 445.294621 |
| pythia_410m | 0.000000 | 0.000000 | 0.000000 |

Four zero optima are certified to floating precision. Pythia1.4B's zero is not certified but at most11.84g remains. The other positive cases retain bounded search gaps; LP ceilings need not be attainable with integral hourly actions.

All-nine actual fixed-run slack, throughput/power/J-token ranges and numeric bounds are in the [opportunity report](../runs/measured_models_independent_audit/opportunity_report.md). Actual slack spans2.347h for GPT2 to414.953h for Llama3B; nominal25% is not actual slack because of deadline rounding and fixed-action selection.

Controlled sensitivity enumerates all zero/one-switch schedules for90 cases, separately changing external offset(0/6/12/18h), deadline multiplier(1/1.25/1.5), or decision interval(15min/1h/4h). These are feasible restricted-search results, **not full multi-switch optima**. At offset+12h, previously baseline-zero Pythia410M achieves1.71661%, GPT2-medium0.22123%, and Pythia1.4B0.20763%. Baseline certificates therefore do not generalize across starts. More slack produces no extra one-switch improvement in the tested1.25→1.5 change; that does not establish unrestricted slack insensitivity. Decision-grid changes have small but observable restricted effects. No pausing or optimized delayed start is used.

Historical2.587716% was a conditional mean of per-episode percentages over50/64 feasible dynamic episodes, not the same workload/start/menu or statistic. Current-menu one-assumption experiments and exact missing inputs are documented. The historical strategy profile is absent locally and from the historical Git tree, preventing exact menu/cost reproduction and causal attribution. Changing only current GPT2 deadline to26h gives no one-switch gain; changing only tokens to8B with9h is infeasible. These do not reproduce the historical result.

## Recommended next experiment

Use exact or bound-certified search for these small measured menus. Repeat selected frontier actions with synchronized serialized power/timing windows, longer stable measurements and randomized run order; include the Pythia nonconverged action. Report uncertainty and GPU-only scope. Recover authoritative trace source, grid-zone mapping and timezone before a regional carbon claim. Then sample externally fixed start times and test realistic switching costs. Equal token count/global batch is the modeled workload criterion, not evidence of equal final training quality.

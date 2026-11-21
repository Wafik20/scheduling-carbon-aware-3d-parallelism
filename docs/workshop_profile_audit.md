# 8-GPU sweep read-only audit

Scope: manifest.json, matrix.csv, results.csv, all case statuses, failed launcher logs, calibration existence and recorded rank mappings. No jobs run or source edits.

## Coverage
- 364 manifest/matrix placements; 348 unique completed results, 16 failed, no unattempted cases. All failures have return code 143; termination alone is not a root-cause diagnosis.
- 166 unique model/TP/PP/DP/microbatch tasks, 162 with at least one successful placement.
- Standard order means `tp-pp-dp in equivalent_orders`, not literal order equality. 166 expected standard rows: 157 completed (52 literal tp-pp-dp, 105 tp-dp-pp aliases); 9 failed.

| Model | All completed/expected | Standard completed/expected |
|---|---:|---:|
| gpt2 | 38/38 | 16/16 |
| gpt2_medium | 40/42 | 19/20 |
| gpt2_large | 38/38 | 16/16 |
| pythia_410m | 40/42 | 18/20 |
| pythia_1_4b | 39/42 | 19/20 |
| pythia_2_8b | 38/42 | 17/20 |
| llama_1b | 41/42 | 20/20 |
| llama_2b | 36/38 | 15/16 |
| llama_3b | 38/40 | 17/18 |

## Missing results (all failed)
| Case ID | Standard? | Launcher evidence |
|---|---|---|
| gpt2_medium_tp1_pp8_dp1_mbs2_tp-dp-pp | yes | Bus error, Profiler exited with rc=1 |
| gpt2_medium_tp2_pp2_dp2_mbs2_dp-tp-pp | no | Profiler exited with rc=1 |
| pythia_410m_tp1_pp2_dp4_mbs2_tp-pp-dp | yes | Bus error, Profiler exited with rc=1 |
| pythia_410m_tp4_pp1_dp2_mbs2_tp-dp-pp | yes | Profiler exited with rc=1 |
| pythia_1_4b_tp1_pp2_dp4_mbs1_tp-dp-pp | no | Profiler exited with rc=1 |
| pythia_1_4b_tp8_pp1_dp1_mbs1_tp-dp-pp | yes | Profiler exited with rc=1 |
| pythia_1_4b_tp2_pp2_dp2_mbs2_dp-tp-pp | no | DistBackendError, Profiler exited with rc=1 |
| pythia_2_8b_tp4_pp1_dp2_mbs1_tp-dp-pp | yes | Profiler exited with rc=1 |
| pythia_2_8b_tp1_pp1_dp8_mbs2_tp-dp-pp | yes | out of memory, Profiler exited with rc=1 |
| pythia_2_8b_tp1_pp2_dp4_mbs2_tp-pp-dp | yes | Profiler exited with rc=1 |
| pythia_2_8b_tp2_pp4_dp1_mbs2_dp-pp-tp | no | Profiler exited with rc=1 |
| llama_1b_tp2_pp2_dp2_mbs2_pp-tp-dp | no | DistBackendError, Profiler exited with rc=1 |
| llama_2b_tp2_pp2_dp2_mbs2_dp-pp-tp | no | Profiler exited with rc=1 |
| llama_2b_tp4_pp2_dp1_mbs2_tp-dp-pp | yes | Profiler exited with rc=1 |
| llama_3b_tp1_pp1_dp8_mbs2_tp-dp-pp | yes | Profiler exited with rc=1 |
| llama_3b_tp1_pp4_dp2_mbs2_tp-dp-pp | no | Profiler exited with rc=1 |

## Metric validation and warnings
- All 348 rows have finite positive power, throughput, step time, step energy, and joules/token; all 348 calibration paths exist.
- `energy_per_step_j == power_w * step_time_s` exactly for all rows. Max relative error in `throughput_tok_s * step_time_s / tokens_per_iter` is 0.0274%; max relative difference between joules/token and power/throughput is 0.00986% (consistent with stored rounding).
- Ranges: power 624.1–1601.3 W; throughput 799.77–103940.93 tokens/s; step 0.1576–5.1215 s; energy 120.37488–3651.912 J/step.
- 33/348 successful warmups did not converge; standard subset 20/157. Completed is not identical to stable calibration.
- IMPORTANT: results have 233 rows at sequence length 1024 / 16384 tokens per iteration and 115 at length 256 / 4096 tokens per iteration. All global batches are 16. Do not normalize every row with 16384 tokens.
- 0 results differ from manifest sequence length; by model: {}. Treat actual results/calibration lengths as authoritative and flag mixed workloads.
- Every recorded result has eight unique GPU UUIDs and eight rank mappings with TP/PP/DP groups found in its manifest groups. This is recorded-data checking, not a new runtime validation.
- validation/fork-validation.json passed CPU-only AST-isolated group checks; explicitly excludes CUDA/NCCL, model construction and measured rank/device validation. implementation-summary.json slurm_submitted=false is historical preparation metadata, not current sweep completion state.
- 14 model/config/microbatch combinations excluded before manifest generation due to layer/head/query-group divisibility; these are unsupported exclusions, not failed runs.

## Standard rows (successful unique configurations)
| Case ID | Actual seq | W | tok/s | s/step | J/step | Warmup converged |
|---|---:|---:|---:|---:|---:|---|
| gpt2_tp1_pp1_dp8_mbs1_tp-dp-pp | 1024 | 706.7 | 61205.97 | 0.2677 | 189.18359 | True |
| gpt2_tp1_pp2_dp4_mbs1_tp-pp-dp | 1024 | 737.9 | 64831.24 | 0.2527 | 186.46733 | True |
| gpt2_tp1_pp4_dp2_mbs1_tp-pp-dp | 1024 | 700.6 | 70827.04 | 0.2313 | 162.04878 | True |
| gpt2_tp2_pp1_dp4_mbs1_tp-dp-pp | 1024 | 679.9 | 51750.65 | 0.3166 | 215.25634 | True |
| gpt2_tp2_pp2_dp2_mbs1_tp-pp-dp | 1024 | 674.4 | 53444.0 | 0.3066 | 206.77104 | True |
| gpt2_tp2_pp4_dp1_mbs1_tp-dp-pp | 1024 | 664.5 | 53893.23 | 0.304 | 202.00800 | True |
| gpt2_tp4_pp1_dp2_mbs1_tp-dp-pp | 1024 | 670.2 | 35552.66 | 0.4608 | 308.82816 | True |
| gpt2_tp4_pp2_dp1_mbs1_tp-dp-pp | 1024 | 658.0 | 32825.94 | 0.4991 | 328.40780 | True |
| gpt2_tp1_pp1_dp8_mbs2_tp-dp-pp | 1024 | 743.1 | 73899.66 | 0.2217 | 164.74527 | True |
| gpt2_tp1_pp2_dp4_mbs2_tp-pp-dp | 1024 | 742.8 | 81392.42 | 0.2013 | 149.52564 | True |
| gpt2_tp1_pp4_dp2_mbs2_tp-pp-dp | 1024 | 754.9 | 94191.75 | 0.1739 | 131.27711 | True |
| gpt2_tp2_pp1_dp4_mbs2_tp-dp-pp | 1024 | 749.1 | 80633.2 | 0.2032 | 152.21712 | True |
| gpt2_tp2_pp2_dp2_mbs2_tp-pp-dp | 1024 | 751.4 | 84460.09 | 0.194 | 145.77160 | True |
| gpt2_tp2_pp4_dp1_mbs2_tp-dp-pp | 1024 | 745.9 | 87523.86 | 0.1872 | 139.63248 | True |
| gpt2_tp4_pp1_dp2_mbs2_tp-dp-pp | 1024 | 714.6 | 49695.61 | 0.3297 | 235.60362 | True |
| gpt2_tp4_pp2_dp1_mbs2_tp-dp-pp | 1024 | 696.8 | 46091.05 | 0.3555 | 247.71240 | True |
| gpt2_medium_tp1_pp1_dp8_mbs1_tp-dp-pp | 1024 | 707.4 | 24575.93 | 0.6667 | 471.62358 | True |
| gpt2_medium_tp1_pp2_dp4_mbs1_tp-pp-dp | 1024 | 725.5 | 30172.32 | 0.543 | 393.94650 | True |
| gpt2_medium_tp1_pp4_dp2_mbs1_tp-pp-dp | 1024 | 748.7 | 37798.92 | 0.4335 | 324.56145 | True |
| gpt2_medium_tp1_pp8_dp1_mbs1_tp-dp-pp | 1024 | 771.0 | 45178.46 | 0.3627 | 279.64170 | True |
| gpt2_medium_tp2_pp1_dp4_mbs1_tp-dp-pp | 1024 | 696.6 | 23848.72 | 0.687 | 478.56420 | True |
| gpt2_medium_tp2_pp2_dp2_mbs1_tp-pp-dp | 1024 | 702.0 | 26565.64 | 0.6167 | 432.92340 | True |
| gpt2_medium_tp2_pp4_dp1_mbs1_tp-dp-pp | 1024 | 719.4 | 31175.0 | 0.5255 | 378.04470 | True |
| gpt2_medium_tp4_pp1_dp2_mbs1_tp-dp-pp | 1024 | 686.0 | 16769.74 | 0.977 | 670.22200 | True |
| gpt2_medium_tp4_pp2_dp1_mbs1_tp-dp-pp | 1024 | 685.9 | 17254.48 | 0.9496 | 651.33064 | True |
| gpt2_medium_tp8_pp1_dp1_mbs1_tp-dp-pp | 1024 | 646.7 | 9410.34 | 1.7411 | 1125.96937 | True |
| gpt2_medium_tp1_pp1_dp8_mbs2_tp-dp-pp | 1024 | 722.2 | 28310.21 | 0.5787 | 417.93714 | True |
| gpt2_medium_tp1_pp2_dp4_mbs2_tp-pp-dp | 1024 | 747.6 | 36673.62 | 0.4468 | 334.02768 | True |
| gpt2_medium_tp1_pp4_dp2_mbs2_tp-pp-dp | 1024 | 808.9 | 48952.43 | 0.3347 | 270.73883 | True |
| gpt2_medium_tp2_pp1_dp4_mbs2_tp-dp-pp | 1024 | 752.8 | 34184.81 | 0.4793 | 360.81704 | True |
| gpt2_medium_tp2_pp2_dp2_mbs2_tp-pp-dp | 1024 | 783.3 | 41389.9 | 0.3958 | 310.03014 | True |
| gpt2_medium_tp2_pp4_dp1_mbs2_tp-dp-pp | 1024 | 813.0 | 48325.94 | 0.339 | 275.60700 | True |
| gpt2_medium_tp4_pp1_dp2_mbs2_tp-dp-pp | 1024 | 705.4 | 19057.59 | 0.8597 | 606.43238 | True |
| gpt2_medium_tp4_pp2_dp1_mbs2_tp-dp-pp | 1024 | 693.0 | 18235.61 | 0.8985 | 622.66050 | True |
| gpt2_medium_tp8_pp1_dp1_mbs2_tp-dp-pp | 1024 | 665.8 | 11107.89 | 1.475 | 982.05500 | True |
| gpt2_large_tp1_pp1_dp8_mbs1_tp-dp-pp | 1024 | 705.3 | 12109.68 | 1.353 | 954.27090 | True |
| gpt2_large_tp1_pp2_dp4_mbs1_tp-pp-dp | 1024 | 730.8 | 16413.02 | 0.9982 | 729.48456 | True |
| gpt2_large_tp1_pp4_dp2_mbs1_tp-pp-dp | 1024 | 789.7 | 23100.64 | 0.7092 | 560.05524 | True |
| gpt2_large_tp2_pp1_dp4_mbs1_tp-dp-pp | 1024 | 723.5 | 13608.05 | 1.204 | 871.09400 | True |
| gpt2_large_tp2_pp2_dp2_mbs1_tp-pp-dp | 1024 | 758.1 | 17107.64 | 0.9577 | 726.03237 | True |
| gpt2_large_tp2_pp4_dp1_mbs1_tp-dp-pp | 1024 | 785.5 | 20023.16 | 0.8183 | 642.77465 | True |
| gpt2_large_tp4_pp1_dp2_mbs1_tp-dp-pp | 1024 | 718.5 | 10052.34 | 1.6299 | 1171.08315 | True |
| gpt2_large_tp4_pp2_dp1_mbs1_tp-dp-pp | 1024 | 715.6 | 10506.07 | 1.5595 | 1115.97820 | True |
| gpt2_large_tp1_pp1_dp8_mbs2_tp-dp-pp | 1024 | 710.5 | 13527.3 | 1.2112 | 860.55760 | True |
| gpt2_large_tp1_pp2_dp4_mbs2_tp-pp-dp | 1024 | 756.2 | 19373.55 | 0.8457 | 639.51834 | True |
| gpt2_large_tp1_pp4_dp2_mbs2_tp-pp-dp | 1024 | 849.7 | 28422.73 | 0.5764 | 489.76708 | True |
| gpt2_large_tp2_pp1_dp4_mbs2_tp-dp-pp | 1024 | 767.9 | 17849.85 | 0.9179 | 704.85541 | True |
| gpt2_large_tp2_pp2_dp2_mbs2_tp-pp-dp | 1024 | 832.8 | 23767.9 | 0.6893 | 574.04904 | True |
| gpt2_large_tp2_pp4_dp1_mbs2_tp-dp-pp | 1024 | 904.7 | 32034.52 | 0.5114 | 462.66358 | True |
| gpt2_large_tp4_pp1_dp2_mbs2_tp-dp-pp | 1024 | 722.3 | 10291.16 | 1.592 | 1149.90160 | True |
| gpt2_large_tp4_pp2_dp1_mbs2_tp-dp-pp | 1024 | 720.4 | 10343.59 | 1.584 | 1141.11360 | True |
| pythia_410m_tp1_pp1_dp8_mbs1_tp-dp-pp | 1024 | 756.0 | 22276.38 | 0.7355 | 556.03800 | True |
| pythia_410m_tp1_pp2_dp4_mbs1_tp-pp-dp | 1024 | 794.2 | 30667.05 | 0.5343 | 424.34106 | True |
| pythia_410m_tp1_pp4_dp2_mbs1_tp-pp-dp | 1024 | 876.9 | 39952.85 | 0.4101 | 359.61669 | True |
| pythia_410m_tp1_pp8_dp1_mbs1_tp-dp-pp | 1024 | 983.5 | 54588.3 | 0.3001 | 295.14835 | True |
| pythia_410m_tp2_pp1_dp4_mbs1_tp-dp-pp | 1024 | 748.6 | 22016.0 | 0.7442 | 557.10812 | True |
| pythia_410m_tp2_pp2_dp2_mbs1_tp-pp-dp | 1024 | 790.6 | 28231.61 | 0.5803 | 458.78518 | True |
| pythia_410m_tp2_pp4_dp1_mbs1_tp-dp-pp | 1024 | 821.2 | 31847.85 | 0.5144 | 422.42528 | True |
| pythia_410m_tp4_pp1_dp2_mbs1_tp-dp-pp | 1024 | 740.3 | 16723.96 | 0.9797 | 725.27191 | True |
| pythia_410m_tp4_pp2_dp1_mbs1_tp-dp-pp | 1024 | 738.3 | 17304.38 | 0.9468 | 699.02244 | True |
| pythia_410m_tp8_pp1_dp1_mbs1_tp-dp-pp | 1024 | 712.1 | 9391.24 | 1.7446 | 1242.32966 | True |
| pythia_410m_tp1_pp1_dp8_mbs2_tp-dp-pp | 1024 | 764.5 | 25725.57 | 0.6369 | 486.91005 | True |
| pythia_410m_tp1_pp4_dp2_mbs2_tp-pp-dp | 1024 | 884.7 | 48759.75 | 0.336 | 297.25920 | True |
| pythia_410m_tp1_pp8_dp1_mbs2_tp-dp-pp | 1024 | 1029.1 | 66382.66 | 0.2468 | 253.98188 | True |
| pythia_410m_tp2_pp1_dp4_mbs2_tp-dp-pp | 1024 | 822.4 | 31202.83 | 0.5251 | 431.84224 | True |
| pythia_410m_tp2_pp2_dp2_mbs2_tp-pp-dp | 1024 | 929.2 | 43046.35 | 0.3806 | 353.65352 | True |
| pythia_410m_tp2_pp4_dp1_mbs2_tp-dp-pp | 1024 | 994.0 | 54513.14 | 0.3006 | 298.79640 | True |
| pythia_410m_tp4_pp2_dp1_mbs2_tp-dp-pp | 1024 | 737.0 | 18292.76 | 0.8957 | 660.13090 | True |
| pythia_410m_tp8_pp1_dp1_mbs2_tp-dp-pp | 1024 | 691.7 | 11040.86 | 1.4839 | 1026.41363 | True |
| pythia_1_4b_tp1_pp1_dp8_mbs1_tp-dp-pp | 1024 | 723.7 | 7800.41 | 2.1004 | 1520.05948 | True |
| pythia_1_4b_tp1_pp2_dp4_mbs1_tp-pp-dp | 1024 | 770.5 | 11823.16 | 1.3858 | 1067.75890 | True |
| pythia_1_4b_tp1_pp4_dp2_mbs1_tp-pp-dp | 1024 | 946.9 | 19845.29 | 0.8256 | 781.76064 | True |
| pythia_1_4b_tp1_pp8_dp1_mbs1_tp-dp-pp | 1024 | 1328.2 | 40190.4 | 0.4077 | 541.50714 | True |
| pythia_1_4b_tp2_pp1_dp4_mbs1_tp-dp-pp | 1024 | 775.8 | 10538.91 | 1.5546 | 1206.05868 | True |
| pythia_1_4b_tp2_pp2_dp2_mbs1_tp-pp-dp | 1024 | 946.3 | 17823.49 | 0.9192 | 869.83896 | True |
| pythia_1_4b_tp2_pp4_dp1_mbs1_tp-dp-pp | 1024 | 1211.4 | 31921.77 | 0.5133 | 621.81162 | True |
| pythia_1_4b_tp4_pp1_dp2_mbs1_tp-dp-pp | 1024 | 770.5 | 8179.82 | 2.003 | 1543.31150 | True |
| pythia_1_4b_tp4_pp2_dp1_mbs1_tp-dp-pp | 1024 | 802.1 | 9610.12 | 1.7049 | 1367.50029 | True |
| pythia_1_4b_tp1_pp1_dp8_mbs2_tp-dp-pp | 1024 | 705.6 | 7946.32 | 2.0618 | 1454.80608 | True |
| pythia_1_4b_tp1_pp2_dp4_mbs2_tp-pp-dp | 1024 | 743.3 | 12163.04 | 1.347 | 1001.22510 | True |
| pythia_1_4b_tp1_pp4_dp2_mbs2_tp-pp-dp | 1024 | 908.2 | 20708.74 | 0.7912 | 718.56784 | True |
| pythia_1_4b_tp1_pp8_dp1_mbs2_tp-dp-pp | 1024 | 1296.5 | 41971.64 | 0.3904 | 506.15360 | True |
| pythia_1_4b_tp2_pp1_dp4_mbs2_tp-dp-pp | 1024 | 817.6 | 11886.07 | 1.3784 | 1126.97984 | True |
| pythia_1_4b_tp2_pp2_dp2_mbs2_tp-pp-dp | 1024 | 995.2 | 21210.09 | 0.7725 | 768.79200 | True |
| pythia_1_4b_tp2_pp4_dp1_mbs2_tp-dp-pp | 1024 | 1382.3 | 42393.42 | 0.3865 | 534.25895 | True |
| pythia_1_4b_tp4_pp1_dp2_mbs2_tp-dp-pp | 1024 | 761.8 | 8434.93 | 1.9424 | 1479.72032 | True |
| pythia_1_4b_tp4_pp2_dp1_mbs2_tp-dp-pp | 1024 | 782.1 | 9594.58 | 1.7076 | 1335.51396 | True |
| pythia_1_4b_tp8_pp1_dp1_mbs2_tp-dp-pp | 1024 | 721.2 | 5669.32 | 2.8899 | 2084.19588 | False |
| pythia_2_8b_tp1_pp1_dp8_mbs1_tp-dp-pp | 1024 | 713.2 | 3993.89 | 4.1023 | 2925.76036 | False |
| pythia_2_8b_tp1_pp2_dp4_mbs1_tp-pp-dp | 1024 | 769.1 | 6165.67 | 2.6573 | 2043.72943 | False |
| pythia_2_8b_tp1_pp4_dp2_mbs1_tp-pp-dp | 1024 | 965.3 | 11111.9 | 1.4745 | 1423.33485 | True |
| pythia_2_8b_tp1_pp8_dp1_mbs1_tp-dp-pp | 1024 | 1559.1 | 26273.65 | 0.6236 | 972.25476 | True |
| pythia_2_8b_tp2_pp1_dp4_mbs1_tp-dp-pp | 1024 | 839.1 | 5840.6 | 2.8052 | 2353.84332 | False |
| pythia_2_8b_tp2_pp2_dp2_mbs1_tp-pp-dp | 1024 | 1046.2 | 10401.53 | 1.5752 | 1647.97424 | True |
| pythia_2_8b_tp2_pp4_dp1_mbs1_tp-dp-pp | 1024 | 1601.3 | 22783.26 | 0.7191 | 1151.49483 | True |
| pythia_2_8b_tp4_pp2_dp1_mbs1_tp-dp-pp | 1024 | 861.3 | 5901.09 | 2.7764 | 2391.31332 | False |
| pythia_2_8b_tp8_pp1_dp1_mbs1_tp-dp-pp | 1024 | 763.2 | 3424.03 | 4.785 | 3651.91200 | False |
| pythia_2_8b_tp1_pp4_dp2_mbs2_tp-pp-dp | 1024 | 904.6 | 11409.74 | 1.436 | 1299.00560 | True |
| pythia_2_8b_tp1_pp8_dp1_mbs2_tp-dp-pp | 1024 | 1371.8 | 25087.41 | 0.6531 | 895.92258 | True |
| pythia_2_8b_tp2_pp1_dp4_mbs2_tp-dp-pp | 1024 | 808.2 | 6155.48 | 2.6617 | 2151.18594 | False |
| pythia_2_8b_tp2_pp2_dp2_mbs2_tp-pp-dp | 1024 | 1002.2 | 11296.35 | 1.4504 | 1453.59088 | True |
| pythia_2_8b_tp2_pp4_dp1_mbs2_tp-dp-pp | 1024 | 1516.7 | 25393.05 | 0.6452 | 978.57484 | True |
| pythia_2_8b_tp4_pp1_dp2_mbs2_tp-dp-pp | 1024 | 796.6 | 4824.82 | 3.3958 | 2705.09428 | False |
| pythia_2_8b_tp4_pp2_dp1_mbs2_tp-dp-pp | 1024 | 839.4 | 5772.56 | 2.8383 | 2382.46902 | False |
| pythia_2_8b_tp8_pp1_dp1_mbs2_tp-dp-pp | 1024 | 756.0 | 3436.64 | 4.7674 | 3604.15440 | False |
| llama_1b_tp1_pp1_dp8_mbs1_tp-dp-pp | 256 | 646.6 | 1958.6 | 2.0913 | 1352.23458 | True |
| llama_1b_tp1_pp2_dp4_mbs1_tp-pp-dp | 256 | 659.0 | 3044.06 | 1.3456 | 886.75040 | True |
| llama_1b_tp1_pp4_dp2_mbs1_tp-pp-dp | 256 | 720.1 | 5214.61 | 0.7855 | 565.63855 | True |
| llama_1b_tp1_pp8_dp1_mbs1_tp-dp-pp | 256 | 997.7 | 15926.73 | 0.2572 | 256.60844 | True |
| llama_1b_tp2_pp1_dp4_mbs1_tp-dp-pp | 256 | 660.6 | 2836.48 | 1.444 | 953.90640 | True |
| llama_1b_tp2_pp2_dp2_mbs1_tp-pp-dp | 256 | 727.2 | 5381.53 | 0.7611 | 553.47192 | True |
| llama_1b_tp2_pp4_dp1_mbs1_tp-dp-pp | 256 | 913.6 | 12855.82 | 0.3186 | 291.07296 | True |
| llama_1b_tp4_pp1_dp2_mbs1_tp-dp-pp | 256 | 697.0 | 4218.29 | 0.971 | 676.78700 | True |
| llama_1b_tp4_pp2_dp1_mbs1_tp-dp-pp | 256 | 777.0 | 7355.32 | 0.5569 | 432.71130 | True |
| llama_1b_tp8_pp1_dp1_mbs1_tp-dp-pp | 256 | 710.5 | 4313.19 | 0.9496 | 674.69080 | True |
| llama_1b_tp1_pp1_dp8_mbs2_tp-dp-pp | 256 | 624.1 | 2000.54 | 2.0474 | 1277.78234 | True |
| llama_1b_tp1_pp2_dp4_mbs2_tp-pp-dp | 256 | 645.7 | 3298.94 | 1.2416 | 801.70112 | True |
| llama_1b_tp1_pp4_dp2_mbs2_tp-pp-dp | 256 | 681.6 | 5515.82 | 0.7426 | 506.15616 | True |
| llama_1b_tp1_pp8_dp1_mbs2_tp-dp-pp | 256 | 984.9 | 20596.6 | 0.1989 | 195.89661 | True |
| llama_1b_tp2_pp1_dp4_mbs2_tp-dp-pp | 256 | 636.9 | 3097.41 | 1.3224 | 842.23656 | True |
| llama_1b_tp2_pp2_dp2_mbs2_tp-pp-dp | 256 | 708.3 | 6229.32 | 0.6575 | 465.70725 | True |
| llama_1b_tp2_pp4_dp1_mbs2_tp-dp-pp | 256 | 1024.4 | 20932.88 | 0.1957 | 200.47508 | True |
| llama_1b_tp4_pp1_dp2_mbs2_tp-dp-pp | 256 | 702.7 | 5332.27 | 0.7682 | 539.81414 | True |
| llama_1b_tp4_pp2_dp1_mbs2_tp-dp-pp | 256 | 857.5 | 12071.12 | 0.3393 | 290.94975 | True |
| llama_1b_tp8_pp1_dp1_mbs2_tp-dp-pp | 256 | 771.6 | 7696.88 | 0.5322 | 410.64552 | True |
| llama_2b_tp1_pp1_dp8_mbs1_tp-dp-pp | 256 | 632.5 | 1110.31 | 3.6891 | 2333.35575 | False |
| llama_2b_tp1_pp2_dp4_mbs1_tp-pp-dp | 256 | 658.8 | 1764.65 | 2.3211 | 1529.14068 | False |
| llama_2b_tp1_pp4_dp2_mbs1_tp-pp-dp | 256 | 732.4 | 3307.68 | 1.2383 | 906.93092 | True |
| llama_2b_tp2_pp1_dp4_mbs1_tp-dp-pp | 256 | 661.5 | 1624.72 | 2.5211 | 1667.70765 | False |
| llama_2b_tp2_pp2_dp2_mbs1_tp-pp-dp | 256 | 728.1 | 3016.08 | 1.3581 | 988.83261 | False |
| llama_2b_tp2_pp4_dp1_mbs1_tp-dp-pp | 256 | 933.5 | 7312.26 | 0.5602 | 522.94670 | True |
| llama_2b_tp4_pp1_dp2_mbs1_tp-dp-pp | 256 | 707.1 | 2370.8 | 1.7277 | 1221.65667 | True |
| llama_2b_tp4_pp2_dp1_mbs1_tp-dp-pp | 256 | 804.0 | 4349.53 | 0.9417 | 757.12680 | True |
| llama_2b_tp1_pp1_dp8_mbs2_tp-dp-pp | 256 | 625.4 | 1129.73 | 3.6257 | 2267.51278 | False |
| llama_2b_tp1_pp2_dp4_mbs2_tp-pp-dp | 256 | 636.9 | 1908.38 | 2.1463 | 1366.97847 | True |
| llama_2b_tp1_pp4_dp2_mbs2_tp-pp-dp | 256 | 705.7 | 3482.46 | 1.1762 | 830.04434 | True |
| llama_2b_tp2_pp1_dp4_mbs2_tp-dp-pp | 256 | 652.2 | 1878.64 | 2.1803 | 1421.99166 | True |
| llama_2b_tp2_pp2_dp2_mbs2_tp-pp-dp | 256 | 713.8 | 3645.95 | 1.1234 | 801.88292 | True |
| llama_2b_tp2_pp4_dp1_mbs2_tp-dp-pp | 256 | 1055.1 | 12788.14 | 0.3203 | 337.94853 | True |
| llama_2b_tp4_pp1_dp2_mbs2_tp-dp-pp | 256 | 706.2 | 2918.88 | 1.4033 | 991.01046 | True |
| llama_3b_tp1_pp1_dp8_mbs1_tp-dp-pp | 256 | 634.7 | 799.77 | 5.1215 | 3250.61605 | False |
| llama_3b_tp1_pp2_dp4_mbs1_tp-pp-dp | 256 | 655.4 | 1265.8 | 3.2359 | 2120.80886 | False |
| llama_3b_tp1_pp4_dp2_mbs1_tp-pp-dp | 256 | 738.7 | 2507.28 | 1.6336 | 1206.74032 | True |
| llama_3b_tp2_pp1_dp4_mbs1_tp-dp-pp | 256 | 667.9 | 1277.54 | 3.2062 | 2141.42098 | False |
| llama_3b_tp2_pp2_dp2_mbs1_tp-pp-dp | 256 | 747.6 | 2440.64 | 1.6782 | 1254.62232 | True |
| llama_3b_tp2_pp4_dp1_mbs1_tp-dp-pp | 256 | 1090.9 | 7419.54 | 0.5521 | 602.28589 | True |
| llama_3b_tp4_pp1_dp2_mbs1_tp-dp-pp | 256 | 720.9 | 1965.82 | 2.0836 | 1502.06724 | True |
| llama_3b_tp4_pp2_dp1_mbs1_tp-dp-pp | 256 | 900.6 | 4312.29 | 0.9498 | 855.38988 | True |
| llama_3b_tp8_pp1_dp1_mbs1_tp-dp-pp | 256 | 732.3 | 1927.86 | 2.1246 | 1555.84458 | True |
| llama_3b_tp1_pp2_dp4_mbs2_tp-pp-dp | 256 | 641.2 | 1375.94 | 2.9769 | 1908.78828 | False |
| llama_3b_tp1_pp4_dp2_mbs2_tp-pp-dp | 256 | 691.2 | 2641.52 | 1.5506 | 1071.77472 | True |
| llama_3b_tp2_pp1_dp4_mbs2_tp-dp-pp | 256 | 636.3 | 1307.13 | 3.1336 | 1993.90968 | False |
| llama_3b_tp2_pp2_dp2_mbs2_tp-pp-dp | 256 | 728.7 | 2827.27 | 1.4487 | 1055.66769 | True |
| llama_3b_tp2_pp4_dp1_mbs2_tp-dp-pp | 256 | 1212.5 | 12092.3 | 0.3387 | 410.67375 | True |
| llama_3b_tp4_pp1_dp2_mbs2_tp-dp-pp | 256 | 707.7 | 2236.49 | 1.8314 | 1296.08178 | True |
| llama_3b_tp4_pp2_dp1_mbs2_tp-dp-pp | 256 | 878.9 | 5232.19 | 0.7828 | 688.00292 | True |
| llama_3b_tp8_pp1_dp1_mbs2_tp-dp-pp | 256 | 778.4 | 3149.19 | 1.3007 | 1012.46488 | True |

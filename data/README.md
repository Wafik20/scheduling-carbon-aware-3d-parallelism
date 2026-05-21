# Carbon Trace Data

This directory holds hourly carbon-intensity traces used by the simulator and
smoke tests. The evaluation scripts expect CSVs with:

- `year`
- `month`
- `day`
- `time`
- `carbon` in gCO2/kWh

The helper scripts in `scripts/generate_real_dataset.py`,
`scripts/generate_good_month_traces.py`, and
`scripts/generate_synthetic_dataset.py` can regenerate or extend these traces.
Some helpers use API credentials from `.env`; keep `.env` local.


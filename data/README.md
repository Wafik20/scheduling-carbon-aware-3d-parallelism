# Carbon Trace Data

This directory holds hourly carbon-intensity traces used by the simulator and
smoke tests. The paper uses Electricity Maps carbon-intensity traces for AWS
`us-west-1` in 2024; verify the exact CSV before using a command as a paper
reproduction run.

The evaluation scripts expect CSVs with:

- `year`
- `month`
- `day`
- `time`
- `carbon` in gCO2/kWh

The helper scripts in `scripts/generate_real_dataset.py`,
`scripts/generate_good_month_traces.py`, and
`scripts/generate_synthetic_dataset.py` can regenerate or extend these traces.
Some helpers use API credentials from `.env`; keep `.env` local.

The repository currently includes several 2024 traces and some AWS
region-specific examples. Some file names describe a month or region, while
others are broader aggregate traces used for smoke tests or simulator sweeps.

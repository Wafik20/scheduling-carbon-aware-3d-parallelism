# Carbon trace provenance and recovery

## Exact workshop May recovery

Files under `data/recovered_workshop_may2024/` are byte-for-byte Git blobs from
commit `6c27ee4c2ad30c3b88b2304225be01aa944fa2d2`, original paths `data/<filename>`.
They were recovered with `git show <commit>:<path>` without CSV reserialization.
The current top-level data files were not overwritten. Each recovered file has
744 hourly rows. Units are the simulator's carbon-intensity input (gCO2eq/kWh).

| Filename | SHA-256 | Mean | Minimum | Maximum |
|---|---|---:|---:|---:|
| `carbon_2024_05_hourly.csv` | `8fb0da0d267112b666d726298746f4391804ca619b3924a9a1dec279666db96e` | 142.728495 | 55 | 315 |
| `carbon_2024_05_hourly_aws_us-east-2.csv` | `5857c58e4f8f022908e0b0b524c2e7e48415175c3eea5e00269190fc2ba9df15` | 387.404570 | 308 | 470 |
| `carbon_2024_05_hourly_aws_us-west-1.csv` | `8fb0da0d267112b666d726298746f4391804ca619b3924a9a1dec279666db96e` | 142.728495 | 55 | 315 |
| `carbon_2024_05_hourly_aws_us-west-2.csv` | `626fd4b194dc07ff3c3c2f2e15c8f52f34e37f215b14430532978722ab0acc64` | 74.293011 | 27 | 211 |

The generic May file is exactly identical to the historical `aws_us-west-1`
file. This byte identity does not establish provenance for other generic files.
The historical regional May arrays are mutually different. The historical commit
also contains a generic annual `data/carbon_2024_hourly.csv`; its filename alone
is not evidence of a cloud region, and no regional annual files exist there.

## Current annual files are not regional evidence

The three current `data/carbon_2024_hourly_aws_<region>.csv` files have 8,784 rows
and the same SHA-256:
`a4f7fb2c448b753788f401892eaf7dbd7fec524d50236e4360f0b7f47ef8fdd3`.
Their May arrays differ from each recovered historical regional May array.
These files therefore cannot demonstrate regional differences, despite their names.
Retain them as suspect artifacts, not as verified regional measurements.

## API findings: supported evidence and limits

On 2026-11-01, the public
<https://api.electricitymaps.com/v3/openapi.json> returned HTTP 200 without
credentials. The relevant schema excerpt is saved in
`docs/electricitymaps_past_range_schema_excerpt.json`.

* `/carbon-intensity/past-range` documents `zone`, `lat`, `lon`,
  `disableCallerLookup`, and `temporalGranularity` among its parameters.
* Neither `dataCenterRegion` nor `data_center_region` appears anywhere in that
  specification. Neither provider spelling appears either.
* `disableCallerLookup` defaults to false. Its documented meaning is:
  “If true, the fallback IP-based zone lookup will not be performed.”
* The successful response schema requires top-level `zone`, `data`, and
  `temporalGranularity`.
* The historical generator at `6c27ee4` and the previously present current
  generator both sent `dataCenterRegion` / `dataCenterProvider` and `granularity`.
  This is **not** evidence of a camelCase-to-snake_case regression.

Undocumented/ignored cloud selectors followed by IP-based fallback are a plausible
explanation for identical regional outputs, not a proven account of those
requests. No saved raw response zone metadata ties these existing CSVs to a
server-resolved zone. The current public specification cannot prove whether older
servers accepted cloud selectors. No paid trace requests were made in this audit.

## Retrieval safeguards

The generator now requires `EMAPS_ZONE`, sends the documented `zone`,
`disableCallerLookup=true`, and `temporalGranularity=hourly`, and rejects missing
or mismatched response zones. Obtain and verify the Electricity Maps zone mapping
for the intended AWS region before setting `EMAPS_ZONE`; this change does not
invent an AWS-to-grid mapping. `DATA_CENTER_PROVIDER` / `DATA_CENTER_REGION`
remain filename labels only. Do not relabel arbitrary zones as AWS regions.

It validates hourly count/continuity and finite nonnegative values, checks chunk
size, and writes a `.provenance.json` sidecar with CSV SHA-256, requested cloud
labels/zone, request endpoint/parameters, retrieval time, and response metadata.
Sidecars include requests accumulated in the current invocation (including earlier
months when writing later monthly or annual/seasonal outputs), without authentication
headers or API keys. Existing CSVs without sidecars do not gain provenance from
these code changes. Seasonal filenames now retain provider/region and avoid the
old duplicated `_hourly_hourly` suffix.

Offline verification: `.venv/bin/python -m pytest tests/test_trace_provenance.py -q`.

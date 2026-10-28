# data

| path | contents | in git? |
|---|---|---|
| `raw/series.csv` | weekly EIA series written by `jev-audit fetch` (public data) | yes, so results are reproducible without a key |
| `raw/wpsr_text/YYYY-MM-DD.txt` | optional real WPSR highlights text, one file per week-ending date | your call |
| `processed/` | `states.jsonl` and `labels.csv`, rebuilt by `jev-audit build` | no |
| `demo/` | synthetic data for the offline demo | no |

Week dates are EIA week-ending dates (Fridays). Inventory values are in thousand barrels;
refinery utilization is in percent of operable capacity.

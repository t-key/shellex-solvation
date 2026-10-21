# Identifier renaming (2026-10-21)

Development identifiers were replaced with neutral names in the public data and code.
No numerical value, model parameter, or gate threshold changed. Hashes in
`verification/MEASURED_SCOPE_CHRONOLOGY.json` and `data/splits/solvent_family_design.json`
refer to the archived files with the original identifiers.

| Original | Public |
|---|---|
| `V3_P###` (solvent pair) | `SP###` |
| `V4_PROPOSED_##` (added solute) | `ADD_##` |
| `original_v3` | `original` |
| `new_v4_structural` | `added_structural` |
| `new_v4_scope` | `added_scope` |
| `V2-CAND-<source>-R#` (measured curve) | `<source>-R#` |

In `models/fingerprint_ridge.pkl`, only the `train_ids` list was renamed; the ridge
coefficients and scaler arrays are bitwise identical to the archived file.

After renaming, `reproduce_tables.py` reproduced all 9 tables (97 checks, maximum
error 4.4e-16). `predict.py` on CPU reproduced the three stored parity curves within
5e-7 kcal/mol (tolerance 5e-5) and ran for a new molecule without fallback.

## File renaming (2026-10-21)

| Original | Public |
|---|---|
| `data/measured/measured148_curves.csv` | `data/measured/measured_curves.csv` |
| `results/UNIFAC_APPLICABILITY_148.csv` | `results/UNIFAC_APPLICABILITY.csv` |
| `results/SHELLEX_PAIR_RANKING_200.csv` | `results/SHELLEX_PAIR_RANKING.csv` |
| `models/historical_pair_coefficients.json` | `models/pair_coefficients_train9800.json` |
| `data/molecules/solutes_original.csv` | `data/molecules/solutes.csv` |
| `data/predictions/direct_cosmo_measured.json` | `data/predictions/measured_pred_cosmors.json` |
| `data/predictions/unifac_original_measured.json` | `data/predictions/measured_pred_unifac.json` |
| `data/predictions/earlier_baselines_measured.json` | `data/predictions/measured_pred_baselines.json` |
| `models/backbone/LICENSE-WEIGHTS.md` | removed (identical to `models/LICENSE-WEIGHTS.md`) |

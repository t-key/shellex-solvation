# Release validation

## Numerical integrity

The four backbone/head checkpoints were repackaged to remove historical status
and development bookkeeping. Every model tensor and normalization value was
compared for exact equality before and after packaging; see
`verification/MODEL_NUMERICAL_INTEGRITY.json`. Active gate parameters, labels,
partitions and saved scientific predictions were not retuned or recomputed.
The original source artifact hashes are provenance identifiers, not the hashes
of the repackaged containers. The file manifest identifies the current files.

## Public-only deployment check

A separate copy containing only public files ran the prediction command in
README using RDKit 2026.03.6 and RTX 5070Ti. The test solute was absent from the
176 supplied conformers, so molecular geometry was generated. Scientific arrays
matched the previously verified predictor within 0kcal/mol(tolerance5e-5).
Both seed coefficients and11grid values were finite, endpoints were zero and
the fallback flag matched. Direct CPU/GPU timing is reported separately in `RUNTIME_BENCHMARK.md`. This is a packaging parity check, not an additional
experimental or computed benchmark score. No selection used its output.

Installed dependency packages were used; this was not a fresh operating-system
installation test. An execution audit blocked reads of original research files
outside the copied public payload, with installed dependencies explicitly
allowed. Thus inference did not rely on private weights or original datasets.

## Stored-table reproduction

`python reproduce_tables.py` regenerated9 tables and all97 numeric comparisons
passed. The runtime benchmark is separate from this table-integrity check. Maximum discrepancy was
4.88498e-15; published-table tolerance is0.0005
and the independent stored-vector MAE check uses1e-10. It called no predictor,
QM engine or fitting routine. Regenerated duplicates are created on demand,
not included twice in the distribution.

## Coordinate-version limitation

The original known conformers were generated with RDKit 2026.03.6. An earlier
portability check using2026.03.2 regenerated a different flexible conformer and
changed its prediction. The remedy was to preserve the original fixed inputs
and require the original generator version for genuinely new molecules, not
to alter scientific tolerances or thresholds. Three prior fixed-input parity
checks had maximum error1.1223065664311972e-8kcal/mol. Raw debugging records are
preserved in the author's archive; this release states the limitation and
provides the required coordinates and version.

## Scope and integrity

Text and model metadata contain no original machine paths, missing-source
archive locators or obsolete private-only deployment markers. The source
research files and prior working-release files remained unchanged.
`verification/RELEASE_QC.json` reports counts and checks. Each payload file is
listed in `FILE_MANIFEST.json` and checked by `SHA256SUMS.txt`, with the stated
self-hash exclusions. No new training, reselection, scientific benchmark
scoring or remote upload was performed by this cleanup.

## Original measured scope restored

The original registration primary151 is now reproducible alongside the later
148 OSC analysis and common132 subset. The three restored indomethacin curves
use stored measurements, metrics, predictions and gate flags; no model was
rerun on that reference set. Existing148/common132 values were independently
checked unchanged before updating the canonical tables.

## Current correction checks

Three fixed-input predictor cases were checked against preserved predictions;
all seed/output values and fallback flags passed5e-5kcal/mol tolerance. See
`verification/PREDICTION_PARITY.csv`. The final table script passed97numeric
checks including151rows and independently retained the main148/common132
values. `verification/TABLE_REPRODUCTION.csv` provides the comparison details.
The direct timing benchmark is separate:28runtime queries, no target scoring.

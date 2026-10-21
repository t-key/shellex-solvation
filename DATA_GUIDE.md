# Data and method correspondence

## Calculated data

`data/labels/calculated_curves.csv` contains21,200 curves: training10,640,
validation3,120 and test7,440(earlier sealed test7,200 plus new structures240).
Array columns are JSON arrays. `x_A` is the solvent-A mole fraction and
`excess_kcal_mol` contains the fixed teacher points. `c0`–`c2` are a least-squares
representation, not a replacement for the original point values during error
evaluation. Exact membership and training order are in `data/splits/`.

The original design has100 solutes,56solvents and200solvent pairs; structural
augmentation added six training-label solutes. Broad original split axes are
retained for correspondence. Augmentation changes some family exposure, so the
historical `family` axis does not by itself guarantee a wholly unseen final
model motif. Six withheld solvents remain fixed: o-xylene, anisole, CPME,
limonene, gamma-valerolactone and o-dichlorobenzene. Broad and specific-family
exposure definitions are distinct and saved in `solvent_family_design.json`.

## Why measured151,148 and132 differ

The initial measured-expansion preregistration specified **151 curves/nine
solutes including indomethacin** as primary. The **148 curves/eight solutes**
used in the manuscript main text are a later OSC-scope analysis excluding the
three ionizable indomethacin drug curves. This restriction was defined after
the initial results were observed, then specified in the follow-up model
evaluation plans. It must not be described as an exclusion fixed before the
initial151 scoring or as a new prospective test.

The original151 result is retained for supplementary reporting. The three
additional curves are in `data/measured/out_of_scope_indomethacin.csv`; their
saved predictions, per-method errors and gate diagnostics are included. The
reproduction command regenerates `all151` rows alongside the main148 and
common132 rows. This scope distinction is not a change to any model or gate.

The common132 curves/five solutes are the fixed common-method comparison,
including the original-UNIFAC applicability restriction. UNIFAC does not cover
the three indomethacin curves; its applicability count must not be called151.
Exact memberships are in `data/splits/measured_subsets.json`. A separate
methanol/isooctane phase-concern curve is outside the original primary151.

All151 released measurements retain original-article DOI attribution. The
initial registration, first located post-observed148 diagnostic and later
evaluation registrations are distinguished by hash and time in
`verification/MEASURED_SCOPE_CHRONOLOGY.json`. Filesystem modification times
are explicitly weaker evidence than embedded registration timestamps.
`prior_group` and head-training-overlap strata retain their original meaning;
head overlap does not establish novelty relative to every pretrained source.

## Final method names

| Saved method key | Meaning |
|---|---|
| shellex | Final unweighted two-seed model with frozen distance fallback |
| shellex_no_fallback | Same model before fallback; reported companion |
| shellex_narrow | Earlier ShellEx-narrow comparison |
| shellex_narrow_motifs | Earlier ShellEx-narrow + motifs comparison |
| pair_mean_train10640 | Final training-only pair mean; unseen pair -> zero |
| pair_mean_train9800 | Historical pair mean; saved historical unseen-pair rule |
| fingerprint_2d | Frozen earlier2D fingerprint/ridge baseline |
| zero_excess | Linear interpolation/no excess contribution |
| UNIFAC_original / direct_COSMO | Stored external-baseline results |

Historical curve IDs are stable row identifiers, not selectable active model
versions. `data/predictions/` preserves comparison values needed by the paper;
earlier model checkpoints are not required to aggregate those stored values.

## Tables, maps and files

`results/` contains one canonical copy of each table/figure-data CSV. The
reproduction command creates a separate output folder at runtime and checks
against those canonical values. Computed/experimental per-curve metrics and
two-seed final vectors remain in `data/predictions/` for numerical auditing.

For the200-pair map, teacher bonus=min(0,min grid excess) and penalty=max(0,max
grid excess). Summaries use medians over the defined100-solute and35-solute
seven-family sets; multiplier=exp(-bonus/0.5925). Stored ShellEx map predictions
use untrained test curves, so matched-test-solute rank correlation is distinct
from comparing a partial test subset with the full100/35teacher medians.
Green categories are the original fixed project grouping, not an environmental
certification. See map CSV comparator/scope columns.

`references/data_sources.csv` is the single curve-to-original-article mapping.
Measurement point values and conversion assumptions are described in
`data/measured/README.md`. The two bibliographies distinguish supporting
references from measured source articles.

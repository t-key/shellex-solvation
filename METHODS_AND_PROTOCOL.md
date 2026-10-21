# Final model and analysis protocol

## Prediction target and structure construction

The target is the excess contribution relative to linearly interpolated
pure-solvent endpoints at298.15 K, in kcal/mol. The fixed grid is
[0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 0.9, 0.95, 1]; errors use the interior points unless a table states otherwise.
Three polynomial coefficients give exactly zero excess at both endpoints.

Inputs are neutral molecular graphs. New molecular conformers use deterministic
ETKDGv3 embedding with explicit hydrogens and random coordinates, followed by a
single MMFF optimization (maximum 1000 iterations; convergence required). These
are separate from the DFT structures used for COSMO teacher labels.

Each shell has three rigid solvent copies. All eight ordered assignments of
two solvent identities to those copies are represented with16 poses each
(128 shell records). Intermolecular contact is placed at a minimum2.6angstrom,
with copy/atom mapping and finite-coordinate checks. Configuration weights are
`x_A**n_A * (1-x_A)**n_B / 16`. The final mixed feature moments use x_A=0.5;
the composition dependence of the output is represented by the polynomial.
The rigid shell is a reproducible representation, not a sampled equilibrium
solvation distribution.

## Encoder, heads and fitting

Two fixed MNSol-pretrained PaiNN/field encoders use19 descriptor inputs and
produce combined7168-dimensional features; the full definitions are in
`code/shellex/`. Frozen pure/mixed feature means and variances feed a
7168–256–128–3 head with SiLU activations and dropout0.1. Seeds are13 and29;
predicted excess curves are averaged across seeds before deployment rules.

Final head training used10,640 calculated curves and validation3,120 curves.
Training feature standardization was fitted on training data. The unweighted
interior-point mean squared loss used AdamW(lr0.001, weight_decay0.01), maximum
400 epochs, validation every10 epochs, patience8 validation checks and gradient
clipping5. No experimental target entered this fitting or gate calibration.
The released encoders are frozen; rejected backbone fine-tuning is not the
released model. Optional feature/head training code is supplied separately and
was not run during release preparation.

## Validation-only selection and fixed fallback

Selection used mean-seed solute-balanced validation MAE as the primary metric:
interior-point MAE within curve, then average within solute and across solutes.
The weighted alternative was compared using a1000-resample paired solute-cluster
bootstrap; a difference including zero triggered the predefined slope-nearest-1
tie-break. Unweighted was selected: MAE0.083964524,
slope0.702189100. Weighted-minus-unweighted MAE was
0.002241173,95%CI
[0.001623815,0.002806607].

The deployed distance threshold is14.710166931152344; seed-disagreement
threshold is disabled(null). Distance is the minimum root-mean-square normalized
distance of the1792-dimensional pure feature to the saved training centroids.
An outside prediction uses the training pair mean if available, otherwise zero
excess. Final amplitude scale is1.0; no slope correction is applied. Seed
disagreement is still reported as a diagnostic. `models/gates.json` contains only
the fixed parameters actually used by deployment.

## Baselines and timing interpretation

Zero excess means linear interpolation between endpoints. The final pair mean
averages training RK coefficients per canonical solvent pair using10,640 curves;
an unseen pair returns zero. The historical train9,800 comparison used its own
saved rule, with the training-wide coefficient mean for unseen pairs. These
definitions are distinguished in the comparison keys and tables.

The frozen2D baseline uses radius2 Morgan256-bit fingerprints for the solute and
each solvent, interaction features, saved scaling and ridge coefficients from
the earlier9,800-curve training set. It is never refitted by the reproduction
command. UNIFAC and direct-COSMO results are stored numerical predictions; that
command does not rerun their external engines. UNIFAC-applicable measured data
are132 curves/five solutes.

New-molecule ShellEx inference uses molecular mechanics and a frozen model,
without new QM/COSMO calculations. Teacher-label construction and MNSol
pretraining are separate upstream costs; head-only fitting times are not total
pipeline times. Teacher surfaces used the recorded BP86/def2-TZVP optimization
and def2-TZVPD final-surface path, parameterization24a and primary single
conformers. The release does not claim equilibrium conformer averaging. See
the scoped timing CSVs in `results/` and software notices.

## Experimental interpretation and statistics

Report the original primary151/nine-solute result in SI, the later148/eight-solute
OSC analysis in the main text, and common132/five-solute comparisons as specified
in `DATA_GUIDE.md`. The primary estimand is equal-solute MAE; curve-average MAE
is secondary. Overall paired95% intervals resample solutes1000 times, seed20261014;
per-solute intervals resample curves. Small solute-cluster counts limit interval
interpretation; a single-solute sample cannot estimate across-solute uncertainty.

Earlier experimental results were already observed during model development.
Keeping measurements out of fitting/selection does not make these comparisons
a new prospective blind test. Residual-head training overlap is reported
separately; it is not proof of novelty relative to every pretrained source.

## Frozen data and provenance

All final parameters, numerical labels, partitions and stored predictions were
preserved in this release cleanup. Existing development IDs map to final method
names in `DATA_GUIDE.md`. Original registration records and full source hashes
remain in the author's local archive; this public protocol is a consolidated
description, not a newly registered experiment. The active pre-label solvent
family definitions and original registration hashes are retained in
`data/splits/solvent_family_design.json` without prompt or queue histories.

The148 OSC subset was not fixed before the initial151 evaluation: it arose in
post-observed diagnostics and was subsequently included in follow-up plans.
The supplementary151 results remain reproducible from the restored reference
values, without model calls. The indomethacin gate flags are environment-specific
stored outputs, not a new threshold-validation experiment.

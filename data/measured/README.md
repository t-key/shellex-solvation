# Measured values, conversion and attribution

This directory contains the148-curve/eight-solute main OSC analysis and a
separate three-curve indomethacin reference set, with original-article DOI.
Together they reproduce the original primary151 curves/nine solutes. No publisher PDF, scan or complete source
table is distributed. `measured_curves.csv` holds the analysis curves and
source metadata; `measured_reported_points.csv` preserves reported point values.
`converted_points.csv` preserves full-precision conversion inputs/results for
the independently converted14included molarity curves. It is not an additional
test set. `conversion_sensitivity.csv` reports their numerical checks.

Inclusion was fixed before the initial expansion scoring: neutral aromatic
solute, nonaqueous binary solvent, at least five points including both pure
solvent endpoints, and source temperature298.15±1.1 K(including reported298 K,
298.2 K,25C,298.15 K and299.2 K). Reported temperature wording remains in the
curve table. Source composition is the solvent-only mole fraction. Repeated
observations at identical composition use the prescribed mean mole fraction;
reported observations remain available. Endpoints were not substituted from
other papers to improve scores. See `DATA_GUIDE.md` for151/148/132membership.

Phenanthrene molarity conversion uses reported saturated-solution density:
`x_s = c / [c + (rho_sat - c*M_s)/(z*M_A + (1-z)*M_B)]`, with consistent g/L
and mol/L units. Naphthalene conversion uses additive volumes:
`x_s = c / [c + (1000-c*V_s)/(z*V_A + (1-z)*V_B)]`, V in cm3/mol.
`molar_volume_provenance.json` lists nominal constants and supporting sources.
Not every nominal value is an exact measured298.15Kconstant. A solute-volume
±10% sensitivity is conditional, not a total-error bound; mixture excess
volume, density, temperature and measurement errors remain separate. The
phenanthrene nominal158 cm3/mol comparison is diagnostic, not its primary
density-based conversion. High concentration is not infinite-dilution truth.

All curves use the same frozen concentration-proxy target convention. Saved
`excess_kcal_mol` is the scored target; a log-mole-fraction audit excess must
not silently replace it. Stored direct-COSMO records retain the matching
experimental targets and endpoint/volume corrections. Curvature is continuous;
not all phenanthrene systems are assumed linear. The known methanol/isooctane
phase concern is excluded consistently from the homogeneous-primary set.

Please cite the original article DOI for each reused curve and the curated
ShellEx collection. Author-owned conversion/curation contributions follow
CC BY4.0; that grant does not relicense third-party measurements or publisher
documents. Original rights and public-domain elements remain unaffected.
Six legacy in-press article attributions are scope-based, as explicitly marked
in `references/data_sources.csv`, not newly independently extracted tables.

## Scored excess target

For solvent-only composition x_A and solute mole fraction x_s, the fixed
concentration proxy is `c_proxy = 1000*x_s / Vmix`, with
`Vmix = x_A*V_A + (1-x_A)*V_B` in cm3/mol. Set
`g = -R*T*ln(c_proxy)` using R=8.31446261815324/4184 kcal/(mol K) and the
source temperature T. The scored excess is
`g(x_A) - x_A*g(1) - (1-x_A)*g(0)`.
The frozen QSPR solvent volumes V_A/V_B are retained per curve in
`data/predictions/measured_pred_cosmors.json` together with endpoint terms and
experimental targets. Nominal volumes used to convert measured molarity to
mole fraction are a separate conversion stage. This concentration proxy does
not become an exact finite-concentration or infinite-dilution standard-state
measurement merely by changing units. The frozen ML model has no temperature
input; the experimental accepted window is described above.

## Scope chronology and gate interpretation

The original measured-expansion primary was151, including indomethacin. The148
OSC-scope restriction was defined after those initial results were available
and later specified in follow-up evaluations. See `DATA_GUIDE.md` and the scope
chronology for the distinction. The reference CSV does not add a newly scored
test set: it restores already stored values, including the original article
DOI10.1021/acs.jced.8b00536.

Only one of the three indomethacin solvent environments crossed the fixed
feature-distance threshold14.710166931152344; that curve used zero excess.
The other two retained neural predictions. The distance includes solvent
environment, so this is not a molecule-wide classification or evidence of
reliable general novelty detection. See `results/INDOMETHACIN_GATE_DIAGNOSTICS.csv`.

For the one flagged environment, stored raw MAE was0.5376310833kcal/mol and the
deployed zero-excess MAE was0.8742142995kcal/mol. Thus the flag is a recorded
high-distance case, but the fallback did not improve that curve. It must not
be presented as evidence that this safeguard reliably improves out-of-scope
accuracy. No rule was changed after inspecting this comparison.

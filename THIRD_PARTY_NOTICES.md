# Third-party methods, software and data notices

The root MIT license applies to original ShellEx code and documentation, not to
third-party packages, source data or model weights. No installed packages,
ORCA/openCOSMO-RS executables, MNSol raw database or publisher PDFs are vendored.

## PaiNN implementation and attribution

This release contains the project's PaiNN implementation and field head in
`code/shellex/backbone.py`. SchNetPack source code is not bundled.
Retain the scientific attribution: Schütt, K. T.; Unke, O. T.; Gastegger, M.
Equivariant Message Passing for the Prediction of Tensorial Properties and
Molecular Spectra. ICML 2021, PMLR 139, 9377–9388.
https://proceedings.mlr.press/v139/schutt21a.html

## Runtime dependencies (installed separately)

- PyTorch: upstream BSD-style license and bundled third-party notices:
  https://github.com/pytorch/pytorch/blob/main/LICENSE
- RDKit: upstream BSD license: https://github.com/rdkit/rdkit/blob/master/license.txt
- NumPy: upstream BSD license and dependency notices:
  https://github.com/numpy/numpy/blob/main/LICENSE.txt
- pandas: upstream BSD license: https://github.com/pandas-dev/pandas/blob/main/LICENSE
- SciPy: upstream BSD license and notices: https://github.com/scipy/scipy/blob/main/LICENSE.txt
- scikit-learn: upstream BSD license: https://github.com/scikit-learn/scikit-learn/blob/main/COPYING

The installed version's own license remains controlling. External baseline
regeneration (optional) uses additional upstream packages under their own terms;
the default table command only aggregates stored values and does not run them.
Credit openCOSMO-RS and its stated parameterization for generated teacher labels,
and the original UNIFAC/Dortmund literature for stored baseline results; see
`references/references.bib` and environment snapshots.

## Database and measurement attribution

MNSol 2012: Marenich et al., University of Minnesota, Minneapolis, 2012.
https://doi.org/10.13020/3eks-j059 ; https://comp.chem.umn.edu/mnsol/
Raw MNSol is not distributed. Its database conditions are not replaced by MIT or
CC BY 4.0; model distribution is described separately in `models/`.
Measured curve values retain original-article DOI attribution, as documented in
`data/measured/README.md` and `references/data_sources.csv`.

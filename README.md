# ShellEx

Version 1.0.0 (released 2026-10-21).

ShellEx predicts excess solvation free-energy curves for a neutral solute in a
binary solvent mixture at 298.15 K. The model combines rigid solvent-shell
features from a frozen PaiNN encoder with a two-seed neural-network head and a
fixed feature-distance fallback. Repository name: `shellex-solvation`.

## Install and predict

ShellEx was run with Python 3.10. Create an environment, install PyTorch 2.7.1
for your hardware, then install the pinned dependencies:

```text
python3.10 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# GPU (CUDA 12.8, the recorded configuration)
pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
# or CPU only
pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu

pip install -r environment/requirements.txt
```

See `environment/README.md` for the recorded platform and the RDKit version
requirement. Then predict:

```text
python predict.py --solute "Oc1cccc(F)c1" --solvent_a "CCO" --solvent_b "CCCC" --device auto
```

The JSON output includes the two seed predictions, deployed and no-fallback
curves, coefficients, feature distance and fallback flag. `x_A` is the solvent-A
mole fraction; energy is in kcal/mol. Outputs are **excess terms, not absolute
solubility**. Consistent pure-solvent endpoints are needed for absolute curves.
The polynomial is `x(1-x)[c0+c1(2x-1)+c2(2x-1)^2]`.

## Reproduce the paper tables

```text
python reproduce_tables.py
```

This command aggregates stored predictions and checks them against the canonical
CSVs in `results/`; it performs no model inference, fitting or QM calculation.
It writes nine regenerated tables to `results/reproduced/` and a comparison file
`REPRODUCTION_CHECK.csv`. These generated duplicates are not shipped beforehand.
For lightweight table reproduction only, install NumPy and pandas
(`pip install numpy==2.2.6 pandas==2.3.3`).

## Read the release

- `METHODS_AND_PROTOCOL.md`: final architecture, selection, geometry, baselines
  and analysis boundaries.
- `DATA_GUIDE.md`: calculated partitions, original primary151 / later OSC148 / common132, columns and
  final model-to-comparison mapping.
- `RUNTIME_BENCHMARK.md`: direct CPU/GPU cold, warm and geometry-reuse timings.
- `VALIDATION.md`: numerical equivalence, portable execution, table checks and
  the RDKit conformer-version limitation.
- `code/`, `models/`: predictor and complete checkpoints, learned preprocessing,
  fixed gates and baseline artifacts.
- `data/`: labels, splits, original measurement numbers with source DOIs,
  conversion assumptions, fixed structures and stored predictions, including
  the original151 and later148 OSC-scope distinction.
- `results/`: one canonical copy of paper tables/figure CSVs and timing evidence.
- `references/`: verified supporting bibliography and original measured sources.
- `FILE_MANIFEST.json`, `SHA256SUMS.txt`: payload inventory and integrity checks.

Optional training-reproduction scripts are under `code/train/`. They require
explicit new output paths and are not called by either command above. Do not run
them to reproduce published table values; use the frozen stored predictions.

## Licenses

Original code/documentation: MIT (`LICENSE`). Author-generated numerical data and
fixed structures: CC BY 4.0 (`data/LICENSE-DATA.md`). Weights, scalers, learned
gates and learned baseline artifacts: CC BY-NC 4.0 (`models/LICENSE-WEIGHTS.md`).
Measurements retain original-source attribution and rights; see
`data/measured/README.md`. Third-party notices and MNSol source terms remain
separate. See `LICENSES.md` and `THIRD_PARTY_NOTICES.md`.

## Citation

Citation metadata, including author ORCID identifiers, is in `CITATION.cff`.

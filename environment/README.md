# Execution environment

Recorded platform: Python 3.10.9, NumPy 2.2.6, pandas 2.3.3, PyTorch 2.7.1+cu128 and
RDKit 2026.03.6, on Windows with Ryzen 9 9950X3D (16 cores/32 threads),64 GiB RAM and
RTX 5070Ti. Hardware is reported for timing context, not a prerequisite for table
reproduction. Direct CPU4-thread and GPU timings, including structure generation,
are reported in `../RUNTIME_BENCHMARK.md` and
`../results/TIMING_SHELLEX_END_TO_END.csv`, with cold, warm and geometry-reuse
conditions distinguished.

For the recorded CUDA path, install PyTorch using its official CUDA12.8 wheel
index. `torch==2.7.1` alone does not choose a CUDA wheel. Install the other pinned
dependencies from `requirements.txt`. No installed environment is distributed.

RDKit 2026.03.6 is required when generating a molecule absent from the supplied
176 frozen conformers. An earlier installation of2026.03.2 selected a different
flexible conformer despite deterministic seeds. The release retains the original
coordinates for known identities; it does not change model thresholds to absorb
that difference. See `VALIDATION.md`.

The stored teacher labels used openCOSMO-RS parameterization24a; the recorded
Python package snapshot is opencosmorspy0.0.1. ORCA surface generation is not
called during inference or table aggregation. Reproducing the QM teacher itself
would require upstream software and source surfaces, not only this distribution.

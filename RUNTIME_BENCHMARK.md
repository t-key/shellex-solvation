# Direct ShellEx runtime measurements

The final public checkpoints and fixed gates were measured on naphthalene and
thianthrene(the heaviest solute in the main148 panel). Exact SMILES and solvent
pairs are in `results/TIMING_SHELLEX_END_TO_END.csv`. CPU uses4threads; the GPU is
RTX5070Ti. Weights, thresholds and outputs were not selected using these times.

Each of four solute/device cases used one new-process cold query, five warm
queries with the model resident but all three molecular geometries regenerated,
and one further query with the same solute/A geometry retained and a new B.
All queries generated fresh shells. No precomputed features, molecular files
or frozen conformers were used in the measured prediction path. Model loading
does read the normal conformer inventory before the benchmark clears its in-
memory geometries, so that constructor cost remains in the cold model-load time.

The cold clock begins at worker-script entry before third-party imports and ends
when the first curve is ready; it includes imports, model loading and first GPU
initialization. OS process-spawn overhead is not separately measured. Warm
total is median-of-five query-call wall time and excludes model loading. Stage
values are taken from the same median-total replicate, so their sum is the
reported total; replicate totals are provided. GPU intervals are synchronized.

Geometry time wraps ETKDGv3/MMFF generation and its cache writes. Pose time also
includes canonicalization, descriptors and query preparation. PaiNN-feature time
includes the two encoders, pooling and transfers; head+gate includes both heads,
fallback and output construction. The actual implementation generates128 shell
records(8ordered occupancies x16poses); pure summaries use the first4poses of
the pure configurations, not eight additional shell calculations.

The old feature proxy and cached-head throughput remain historical, separately
labeled in `TIMING_EVIDENCE.csv`. They are not the representative end-to-end
latency. These results describe two measured queries, not a guarantee for every
molecule, pair, machine, CPU thread count or concurrent system load. No QM,
new training, model selection or accuracy benchmark scoring was performed.

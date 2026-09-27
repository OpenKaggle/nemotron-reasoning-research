# Kaggle recipe sources

This directory preserves the first-party, human-authored configuration and
orchestration sources behind nine Kaggle scripts. `notebook-recipes/` holds
two notebook sources whose code cells have no recorded outputs. Each adjacent
`kernel-metadata.json` is a provenance snapshot: it records the original
kernel identifier and dependency references, not a promise that every
dependency remains public or that a run can be reproduced without the
competition's access approval.

## Included

- `kaggle-kernels/`: nine distinct Python recipe sources. The trace-audit
  folder has a small launcher and its reusable audit module; the omitted
  `*-bundled.py` file was a generated packaging product, not an additional
  recipe.
- `../notebook-recipes/`: the `kienngx-notebook` and `oracle-sft-r32`
  sources, each verified to contain zero saved cell outputs.

## Boundary

These files may describe datasets, model artifacts, Kaggle kernel inputs, or
output directory names. They contain no copied input rows, prompts, labels,
answers, predictions, token traces, checkpoints, adapters, or submission
archives. Acquire each dependency from the original publisher and follow the
competition rules and the dependency's own terms. Do not add generated
records, execution outputs, private mappings, or credentials to this tree.

The trace-audit metadata now points to the retained launcher rather than its
previous generated bundled copy. That normalization keeps the source release
internally coherent without changing the recorded kernel identifier or source
dependencies.

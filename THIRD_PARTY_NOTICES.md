# Third-Party Data and Models

The repository's MIT license applies to the software written for this project.
Third-party datasets and model weights retain their original terms.

## TruthfulQA

The 30 questions in `data/stimuli/factual_judgment/questions.json` are a
selected and reformatted subset of
[TruthfulQA](https://github.com/sylinrl/TruthfulQA), commit
`d71c110897f5d31c5d7f309e7bc316c152f6f031`. TruthfulQA is distributed under
the Apache License 2.0; a copy is included as `licenses/Apache-2.0.txt`.

## SycophancyEval

The arguments used in the Preference-Induced Sycophancy Task originate from
[SycophancyEval](https://github.com/meg-tong/sycophancy-eval), commit
`9a1694221e3639887138f61deae344335eca6752`. The upstream repository did not
contain a dataset license when this release was prepared. The tracked stimulus
directory therefore contains only the selection manifest, which links to the
pinned upstream file and records the exact indices, item IDs, and intervention
subset. The complete archival data deposit retains the executed prompt and
chat text in model-output records for reproducibility; those source-derived
excerpts remain subject to the upstream terms and are not covered by this
repository's MIT license. Users can rebuild the local stimulus file with
`analysis/python/prepare_feedback_stimuli.py`.

## Model Weights

Model identifiers are listed in `configs/models.yaml`. Model weights are not
redistributed and remain subject to their providers' licenses and access terms.

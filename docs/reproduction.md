# Reproduction Guide

The release supports three levels of reproduction.

## 1. CPU-only verification

No model weights or evaluator APIs are needed:

```bash
python analysis/python/release_audit.py
python analysis/python/verify_reported_results.py \
  --output results/reproduction_check.json
pytest -q
python -m analysis.figures.build_all
```

This validates manifests, reported estimates, statistical inputs,
figure source data, and the public experiment interfaces.

## 2. Statistical reproduction

Reproduce the original Figure 2 cross- and within-domain coefficients from the
repository root:

```bash
python analysis/python/cross_domain_control.py
```

Run the R analyses:

```bash
Rscript analysis/r/subjective_preference.R
Rscript analysis/r/feedback_induced_sycophancy.R
Rscript analysis/r/arithmetic_answering_verification.R
Rscript analysis/r/cross_model_factual_judgment.R
Rscript analysis/r/reasoning_subordination.R
Rscript analysis/r/stance_taking.R
```

These scripts consume the compact tracked data and overwrite only derived
files under `results/summaries/`. The default Figure 4 command reproduces its
answer-accuracy regressions. Add `--fit-bayesian` in an R environment with
`brms` to refit the Bayesian multinomial reasoning-pattern models.

## 3. Model inference

Every experiment runner supports `--dry-run`; most support `--max-items` and a
reduced `--alpha-values` grid. Start with the commands in the top-level README.
Full reruns require the registered model weights and substantial GPU time.

### Preparing SycophancyEval arguments

Clone the upstream data repository at the frozen commit and prepare the local
stimuli:

```bash
git clone https://github.com/meg-tong/sycophancy-eval.git
git -C sycophancy-eval checkout 9a1694221e3639887138f61deae344335eca6752
python analysis/python/prepare_feedback_stimuli.py \
  --upstream-file sycophancy-eval/datasets/feedback.jsonl
```

The script reconstructs the selected panel from the pinned upstream file and
writes it under the ignored `data/raw_external/` directory.

## Complete Model Outputs

The Git repositories contain the materials, analysis-ready tables, Source Data,
and a browsable set of model outputs. The archival data release contains the
complete target-layer representations, generated responses, evaluator records,
per-item scores, and token-level diagnostics for all reported experiments in a
single package on Zenodo
([doi:10.5281/zenodo.22304165](https://doi.org/10.5281/zenodo.22304165)).
The preparation scripts in `analysis/python/` convert the complete outputs to
the tracked analysis schema.

To rerun the generation-robustness analyses, extract the archive into the code
repository so that `data/raw/model_outputs/` is present, then run:

```bash
python analysis/python/analyze_prompt_spelling.py
python analysis/python/analyze_decoding_temperature.py
python analysis/python/analyze_reasoning_robustness.py
```

The archived run metadata contains checksums calculated from the cleaned
archival JSONL files. These checks validate the released files rather than the
private source paths used when the experiments were run.

No evaluator API call is required to reproduce the archived judge-validation
statistics. API-based reevaluation is outside the archived reproduction path.

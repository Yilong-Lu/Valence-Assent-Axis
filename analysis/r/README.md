# R Analyses

All scripts resolve inputs and outputs relative to the repository root. They
may also be pointed at alternative locations with their documented command-line
arguments.

- `subjective_preference.R`: mixed-effects semantic and option-order analyses,
  item-level slope magnitudes, bootstrap confidence intervals, and permutation
  baselines.
- `feedback_induced_sycophancy.R`: paired assistant-start state tests, paired
  Strong/Weak verdict tests, and intervention-curve binomial mixed models.
- `arithmetic_answering_verification.R`: item-level accuracy slopes and paired
  framing contrasts.
- `cross_model_factual_judgment.R`: model-specific binomial mixed models for
  coherent hallucinations under truth-aligned pressure.
- `reasoning_subordination.R`: Figure 4 answer-accuracy regressions and the
  optional Bayesian multinomial reasoning-pattern models.
- `stance_taking.R`: Figure 5 answer/reasoning stance and Sound-Reasoning
  mixed-effects models.

Install `lme4`, `lmerTest`, `emmeans`, `nlme`, and `jsonlite`, then run each
script from the repository root. Reproducing the Bayesian multinomial models
also requires `brms`. Statistical outputs are ordinary CSV and JSON files under
`results/summaries/`.

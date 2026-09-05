# Figure Source Data

These compact files are the direct inputs to `analysis/figures/`. They contain
the rows or fitted estimates displayed in the manuscript, with no
machine-specific paths.

| File | Figure | Content |
|---|---|---|
| `figure2_layer_profiles_qwen25_14b.json` | 2a | Corrected Binary/Continuous PC1 layer profiles |
| `figure2_pca_qwen25_14b.json` | 2b-c | Selected-layer PCA coordinates and component summaries |
| `figure2_intervention_qwen25_14b.csv` | 2d-e | Reported Value Judgment and Sentiment intervention rows |
| `figure3.csv` | 3 | Panel-ready Qwen2.5-14B control data and intervals |
| `figure4_*.csv` | 4 | Evaluated Alphabetical Order and Factual Judgment rows |
| `figure5_stance_qwen25_14b.csv` | 5b-c | Evaluated answer/reason stance and reasoning categories |
| `figure6.csv` | 6 | Cross-model layer profiles and fitted estimates |
| `supplement_cross_domain_intervention.csv` | Supplementary Information | Value Judgment and Sentiment intervention rows across eight models, with split-qualified statement IDs |
| `supplement_alphabetical_order.csv` | Supplementary Information | Alphabetical Order accuracy and reasoning categories |
| `supplement_factual_judgment.csv` | Supplementary Information | Factual Judgment reasoning categories |
| `supplement_stance_taking.csv` | Supplementary Information | Answer and reasoning stance scores |
| `tables/valence_axis_similarity.csv` | Supplementary Table B3 | Judgment/Valence projection and axis correlations |
| `tables/objective_task_baseline_accuracy.csv` | Supplementary Table B8 | Baseline objective-task accuracy by model |
| `tables/reasoning_subordination_coefficients.csv` | Supplementary Table B9 | Bayesian reasoning-category coefficients |
| `qualitative_examples/example_index.csv` | Figures 4-5 | Stable links from displayed excerpts to selected raw outputs |
| `qualitative_examples/figure4c_qwen25_14b_selected_outputs.json` | Figure 4c | Exact baseline and conflicting-pressure generations |
| `qualitative_examples/figure5a_qwen25_14b_selected_outputs.json` | Figure 5a | Exact oppose, baseline, and support generations |

Figure 5e-g load the analysis-ready feedback rows in
`data/processed/feedback_induced_sycophancy/` directly. The two Supplementary
figures likewise load the public processed data or statistical summaries rather
than another duplicate Source Data export.

The Figure 2 intervention subset can be regenerated from the full deposited
output with `analysis/python/prepare_figure2_source.py`. Figure 3 and Figure 6
combine the archived outputs produced by the statistical scripts under
`analysis/r/` with original-manuscript analysis rows; their checked-in Source
Data tables are the stable plotting interface.

The Supplementary tables can be regenerated from the archived experiment
outputs with:

```bash
python analysis/python/prepare_cross_domain_source.py \
  --steering-root /path/to/archived_steering_outputs
python analysis/python/prepare_supplement_source.py \
  --results-root /path/to/results
```

The archived stance judge files contain a small number of missing scores.
They are retained as missing values in `supplement_stance_taking.csv`, matching
the original plotting behavior; no values are imputed.

The three files under `tables/` transcribe the values in the
Supplementary Information. They are portable table data, not newly fitted analyses.
The qualitative-example index distinguishes the full selected raw responses
from their shortened display excerpts. Figure 4c was recovered from the archived
`alphabetical_think_answer` output, and Figure 5a from the saved
Qwen2.5-14B notebook output at Layer 28. Figure 4f points to the compact raw JSON
already retained in the original public repository.

# Analysis Data

This directory contains the smallest analysis-ready representation of each
reported experiment. It is intentionally distinct from model-native raw output:
generated response text, token IDs, token log probabilities, and transfer
metadata remain in the external archive.

The tables retain the experimental unit used by inference:

- `answer_label_control/`: one row per model and response-label condition.
- `subjective_preference/`: one row per lexical pair and intervention level,
  plus one assistant-start projection row per pair.
- `feedback_induced_sycophancy/`: one row per argument, feedback condition, and
  intervention level; generated reasons are omitted because they were not a
  confirmatory endpoint.
- `arithmetic_answering_verification/`: one item-slope row per arithmetic
  expression and framing condition.
- `generation_robustness/`: compact prompt-spelling and decoding-temperature
  endpoints.
- `valence_representation/`, `single_letter_order/`, `alphabetical_order/`,
  `factual_judgment/`, and `stance_taking/`: task-specific analysis tables.
- `cross_model/`: cross-model layer, intervention, and Factual Judgment tables.

`manifest/results.yaml` records expected row counts and model coverage. The two
`prepare_*_analysis.py` scripts document how the larger external outputs are
reduced to the tracked tables.

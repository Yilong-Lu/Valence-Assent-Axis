# Data Dictionary

## Shared Identifiers

| Field | Meaning |
|---|---|
| `model_name`, `model_key` | Stable key from `configs/models.yaml` |
| `target_layer`, `layer` | Zero-based transformer block receiving the intervention |
| `item_id`, `pair_id`, `QID` | Stable stimulus identifier within a task |
| `alpha_norm` | Piecewise-normalized intervention coefficient in `[-1, 1]` |
| `alpha`, `alpha_raw` | Model-specific raw coefficient applied to the VAA vector |
| `alignment_pressure` | Analysis coordinate oriented toward the registered correct answer |
| `condition`, `experiment`, `mode` | Task condition or prompt format |

## Representation and Candidate Scores

| Field | Meaning |
|---|---|
| `PCA1`, `PCA2` | Item coordinates on the first two task-derived principal components |
| `D_stance` | Projection onto the independently derived VAA direction |
| `vaa_projection_*` | Raw dot product with the VAA |
| `vaa_projection_unit_*` | Projection onto the unit-normalized VAA |
| `logprob_diff_*`, `d_AB`, `d_BA` | Complete candidate-sequence log-probability differences |
| `semantic_component` | `(d_AB + d_BA) / 2`, invariant to display order |
| `position_component` | `(d_AB - d_BA) / 2`, option-order component |

## Generated Responses

| Field | Meaning |
|---|---|
| `answer`, `reason` | Parsed final answer and generated reasoning text |
| `correct` | Indicator that the parsed answer matches the registered answer |
| `response_type` | Judge-derived reasoning category |
| `verdict_valid` | Deterministic Strong/Weak parser found one unambiguous verdict |
| `verdict_strong` | Binary Strong verdict among parsed responses |
| `answer_stance`, `reasoning_stance` | Qwen judge scores on the registered stance scale |

## Analysis Tables

- `data/processed/subjective_preference/trajectories/`: pair-level steering
  trajectories for complete-sequence and first-token analyses.
- `data/processed/subjective_preference/assistant_start_projection/`: unsteered
  AB/BA assistant-start projections.
- `data/processed/feedback_induced_sycophancy/`: item-level feedback and
  intervention outcomes plus descriptive curve summaries.
- `data/processed/arithmetic_answering_verification/item_slopes/`: one slope per
  arithmetic item and prompt mode.
- `data/processed/generation_robustness/`: decoding-temperature and prompt-
  spelling summaries.
- `data/source_data/`: stable panel-ready interfaces. Blank columns are expected
  where a CSV combines panels with different schemas.

Detailed estimands and orientation conventions are defined in
`docs/scientific_methods.md`; file counts and paths are machine-readable in the
three manifests under `manifest/`.

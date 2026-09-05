# Subjective Valence and Factual Assent are Jointly Represented in Large Language Models

This repository contains code, model configuration, VAA vectors, experimental
materials, and analysis resources for the Valence-Assent Axis manuscript.

The repository contains the eight-model registry, selected VAA vector
artifacts, versioned experimental prompts, shared intervention and scoring
utilities, task runners, compact analysis data, and the statistical pipelines
used for the manuscript.

## Quick start

Python 3.10 or newer is required. The commands below install the CPU analysis
stack, verify the reported numerical results, and rebuild all programmatic
figures without loading model weights:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[analysis,test]'
python analysis/python/release_audit.py
python analysis/python/verify_reported_results.py \
  --output results/reproduction_check.json
pytest -q
python -m analysis.figures.build_all
```

See [Installation and Compute](docs/installation.md) for GPU and R setup,
[Reproduction Guide](docs/reproduction.md) for the three reproduction levels,
and [Data Dictionary](docs/data_dictionary.md) for table fields. Scientific
estimands and orientation conventions are documented in
[Scientific Implementation Notes](docs/scientific_methods.md).

## Code and data releases

This repository contains the experiment runners, prompt registry, VAA vectors,
statistical analyses, figure builders, and compact tables needed for routine
reproduction. Use it by itself to inspect the implementation, run experiments,
or rebuild the reported statistics and programmatic figures from the included
analysis tables.

The companion [*Valence-Assent Axis Data*](https://doi.org/10.5281/zenodo.22304165)
record provides the complete research data package, including Source Data,
human evaluations, and full per-item model outputs. Use both releases when
auditing individual responses, rebuilding processed tables, or reusing the data
without installing the experiment package. The data README maps each data
family back to its analysis entry point in this repository.

## Figures and notebooks

Install the analysis dependencies and rebuild all reproducible figure outputs:

```bash
python -m pip install -e '.[analysis]'
python -m analysis.figures.build_all
```

One script and one editable notebook are provided for each main figure.
Supplementary figure code is separated by scientific function. Complete
manuscript figures, Source Data, and the scope of each automated builder are
listed in `manifest/figures.yaml`; see `figures/README.md` for the two figures
whose response-example panels were arranged in a vector editor.

Illustrative intervention examples are available in
[Intervention Examples](docs/examples.md). Compact raw generations and a
browser-based viewer are available under `examples/`.

## Model configuration

Registered model identifiers, target layers, raw intervention ranges, and
artifact paths are defined in `configs/models.yaml`. A local model directory can
be supplied with the model-specific environment variable listed in that file;
otherwise the registered model ID is passed to the model loader.

VAA vectors are stored as normalized, one-dimensional NumPy arrays under
`artifacts/vaa_vectors/`. They can be loaded without pickle:

```python
import numpy as np

vector = np.load(
    "artifacts/vaa_vectors/qwen25_14b/vector_layer28.npy",
    allow_pickle=False,
)
```

Install the package in editable mode before running the experiment entry points:

```bash
python -m pip install -e .
```

## Experimental implementation

Experimental prompt templates are registered in `configs/prompts.json`.

Normalized intervention coefficients are converted to model-specific raw
strengths with `vaa.steering.normalized_alpha_to_raw`. The intervention hook is
persistent: it adds the VAA vector at every sequence position whenever the
selected transformer layer is evaluated, including autoregressive generation
steps.

Control-task candidates are scored with full-sequence log probabilities in
`vaa.scoring`. Candidate text is independently tokenized after leading
whitespace is removed, matching the convention used in the reported
experiments. First-token analyses will be retained only where they were used as
an explicit robustness analysis.

`vaa.models.load_model_bundle` applies these settings consistently and returns
the model, tokenizer, selected VAA vector, and model metadata. The model input
device is left to `device_map` and the calling environment; no GPU assignment is
hard-coded. `vaa.activations.capture_assistant_start_activations` extracts the
target-layer state at the final prompt position after the chat template has
opened the assistant turn. This is the activation endpoint used for the
reported prompt-state analyses.

## Value Judgment and cross-domain control

The Value Judgment task family is configured in
`configs/experiments/judgment_tasks.yaml`. It contains the 134-statement VAA
extraction set, the full 175-item Value Judgment intervention set, and the
matched 175-headline Sentiment Analysis set.

Validate the registered inputs without loading model weights:

```bash
python experiments/value_judgment/build_axis.py --model qwen25_14b --dry-run
python experiments/value_judgment/run_intervention.py --model qwen25_14b --dry-run
```

Rebuild the selected-layer axis or run all four intervention curves:

```bash
python experiments/value_judgment/build_axis.py --model qwen25_14b
python experiments/value_judgment/run_intervention.py --model qwen25_14b
```

Use `--all-layers` with `build_axis.py` to rebuild the Value Judgment PC1 at
every transformer block. Generated files are written under
`results/generated/` by default; this directory is excluded from version
control.

## Valence and Single-Letter Order controls

Validate the fixed 160-word Valence set and the two matched 100-statement
Single-Letter Order conditions without loading model weights:

```bash
python experiments/valence/build_axis.py --model qwen25_14b --dry-run
python experiments/single_letter_order/build_axis.py --model qwen25_14b --dry-run
```

Run the selected-layer representational analyses:

```bash
python experiments/valence/build_axis.py --model qwen25_14b
python experiments/single_letter_order/build_axis.py --model qwen25_14b
```

The Single-Letter runner executes both the original right/wrong condition and
the matched true/false answer-label control by default. Use `--answer-labels`
to select one condition, and `--no-sample-responses` for a faster diagnostic
run that retains candidate probabilities but skips one-token sampling.

## Subjective Preference

The Subjective Preference runner uses the reported content-free prompt, scores
complete candidate sequences in both AB and BA orders, and runs the registered
11-level normalized intervention grid. It also records the unsteered initial
VAA state immediately before response generation:

```bash
python -m experiments.subjective_preference.run \
  --model qwen25_14b --dry-run
python -m experiments.subjective_preference.run \
  --model qwen25_14b
```

For a short environment check, select one pair from each lexical class and
three intervention levels:

```bash
python -m experiments.subjective_preference.run \
  --model qwen25_14b \
  --max-pairs-per-class 1 \
  --alpha-values -1 0 1
```

Generated raw sequence scores, AB/BA components, initial-state projections,
and tokenization audits are written to
`results/generated/subjective_preference/<model>/` by default.

## Arithmetic Answering and Verification

The arithmetic runner presents each of 150 additions in three matched
conditions: a direct numeric question, a true-statement verification prompt,
and a false-statement verification prompt. It compares full-sequence candidate
log probabilities at the registered 11 intervention levels. Candidate accuracy
is the reported outcome; continuous margins are retained as scoring diagnostics.

Validate the registered design without loading model weights:

```bash
python -m experiments.arithmetic_answering_verification.run \
  --model qwen25_14b --dry-run
```

Run the complete experiment or a short environment check:

```bash
python -m experiments.arithmetic_answering_verification.run \
  --model qwen25_14b
python -m experiments.arithmetic_answering_verification.run \
  --model qwen25_14b --max-items 1 --alpha-values -1 0 1
```

Generated sequence scores, item-level slopes, alpha summaries, and candidate
tokenization audits are written to
`results/generated/arithmetic_answering_verification/<model>/`.

## Alphabetical Order, Factual Judgment, and Stance-Taking

The three open-ended generation tasks share one registered decoding protocol:
temperature 0.2 sampling, top-p 1.0, top-k 50, and at most 512 generated
tokens. The intervention remains active throughout autoregressive generation.
Each public run records its explicit random seed, prompt version, decoding
parameters, raw and normalized intervention strengths, raw completion, and
response-parsing diagnostics.

Validate all three designs without loading model weights:

```bash
python -m experiments.alphabetical_order.run --model qwen25_14b --dry-run
python -m experiments.factual_judgment.run --model qwen25_14b --dry-run
python -m experiments.stance_taking.run --model qwen25_14b --dry-run
```

Run the reported 11-level intervention grids:

```bash
python -m experiments.alphabetical_order.run --model qwen25_14b
python -m experiments.factual_judgment.run --model qwen25_14b
python -m experiments.stance_taking.run --model qwen25_14b
```

For a one-item, three-level environment check, add
`--max-items 1 --alpha-values -1 0 1`. Alphabetical Order runs both
Think-then-Answer and Answer-then-Think by default; select one with
`--conditions think_then_answer` or `--conditions answer_then_think`.
Generated rows are written under `results/generated/<task>/<model>/`.

The executed JSON-format prompts are preserved verbatim in the versioned
prompt registry, including the original spelling and punctuation. Strict JSON
validity and extraction of one unambiguous quoted `answer` field are reported
separately. This keeps output-format compliance distinct from the behavioral
answer used in the reported analyses.

## Preference-Induced Sycophancy

This task presents arguments with no stated preference, a statement that the
user likes the argument, or a statement that the user dislikes it. The
alpha-zero run uses all 296 fixed arguments and records both
the assistant-start VAA projection and the final Strong/Weak verdict.

The manuscript calls this the *Preference-Induced Sycophancy* task. For
compatibility with the archived experiments, the code retains the implementation
identifier `feedback_induced_sycophancy` and the configuration display name
`Feedback-Induced Sycophancy Task`; these names refer to the same protocol.

The tracked stimulus directory contains only item identifiers because
SycophancyEval did not declare a dataset license. The complete archival data
deposit retains the exact executed prompts in model-output records; these
source-derived excerpts remain subject to the upstream terms. Prepare the fixed
panel before running this task:

```bash
python analysis/python/prepare_feedback_stimuli.py \
  --upstream-file /path/to/sycophancy-eval/datasets/feedback.jsonl
```

The preparation script reconstructs the exact 296-item panel from the pinned
upstream file and tracked indices. Then run either protocol:

```bash
python -m experiments.feedback_induced_sycophancy.run_feedback_effect \
  --model qwen25_14b --dry-run
python -m experiments.feedback_induced_sycophancy.run_feedback_effect \
  --model qwen25_14b
```

The intervention run uses the prespecified 100-argument subset under the same
three preference conditions and the registered 11-level alpha grid:

```bash
python -m experiments.feedback_induced_sycophancy.run_intervention \
  --model qwen25_14b --dry-run
python -m experiments.feedback_induced_sycophancy.run_intervention \
  --model qwen25_14b
```

For a short environment check, add `--max-items 1 --alpha-values -1 0 1` to
the intervention command. Both runs use greedy generation and write raw
completions, assistant-start projections, parsed verdicts, and parsing status
under `results/generated/feedback_induced_sycophancy/<protocol>/<model>/`.

## Generation robustness checks

The Prompt-Spelling Check compares the original JSON instructions containing
`anwer` with otherwise identical instructions containing `answer`. It uses 30
Alphabetical Order and 30 Factual Judgment items, three normalized intervention
levels, and three seeds at temperature 0.2:

```bash
python -m experiments.generation_robustness.run_prompt_spelling \
  --model qwen25_14b --dry-run
python -m experiments.generation_robustness.run_prompt_spelling \
  --model qwen25_14b
```

The Decoding-Temperature Sensitivity analysis uses the original prompt, five
intervention levels, greedy decoding, and sampled decoding at temperatures 0.2
and 1.0:

```bash
python -m experiments.generation_robustness.run_decoding_temperature \
  --model qwen25_14b --dry-run
python -m experiments.generation_robustness.run_decoding_temperature \
  --model qwen25_14b
```

Both checks are registered for Qwen2.5-7B/14B/32B/72B and Llama-3.1-8B.
Completed generation cells are reused after their item and decoding settings
are validated. Add `--max-items 1 --alpha-values -0.2 0 0.2` and a separate
`--output-dir` for a short environment check.

## Results and statistical reproduction

The release separates three data tiers:

- `data/processed/` contains compact item-level or trajectory-level records
  needed to reproduce the reported statistics.
- `results/summaries/` contains derived estimates and confidence intervals.
- `data/raw_external/` is the expected destination for complete generated
  responses and token-level diagnostics obtained from the external data
  archive. It is excluded from Git.

The machine-readable inventory is `manifest/results.yaml`. Prepare the compact
Subjective Preference and Preference-Induced Sycophancy tables from a downloaded
full-result archive with:

```bash
python analysis/python/prepare_preference_analysis.py \
  --input-root data/raw_external/subjective_preference
python analysis/python/prepare_feedback_analysis.py \
  --input-root data/raw_external/feedback_induced_sycophancy
```

Reproduce the reported cross- and within-domain MixedLM coefficients with:

```bash
python analysis/python/cross_domain_control.py
```

The principal statistical analyses are implemented in R. The commands below use
the repository-relative defaults and write into `results/summaries/`:

```bash
Rscript analysis/r/subjective_preference.R
Rscript analysis/r/feedback_induced_sycophancy.R
Rscript analysis/r/arithmetic_answering_verification.R
Rscript analysis/r/cross_model_factual_judgment.R
Rscript analysis/r/reasoning_subordination.R
Rscript analysis/r/stance_taking.R
```

Required R packages are `lme4`, `lmerTest`, `emmeans`, `nlme`, and `jsonlite`.
The Subjective Preference analysis includes 5,000 bootstrap samples and 5,000
within-pair permutations by default. Add `--fit-bayesian` to the Figure 4
script, in an environment with `brms`, to reproduce the multinomial
reasoning-pattern models.

Run the automated coverage and manuscript-result checks after any data or
analysis change:

```bash
python analysis/python/verify_reported_results.py \
  --output results/reproduction_check.json
pytest -q
```

The prompt-spelling, decoding-temperature, and reasoning-score scripts under
`analysis/python/` consume full outputs from either locally extracted outputs
under `data/raw_external/` or an extracted data archive under
`data/raw/model_outputs/`. The archive includes the resolved per-item reasoning
scores, so these analyses require no evaluator API calls. Compact summaries
used in the reported analyses are included for inspection without downloading
generated text.

## Judge validation

Install the analysis dependencies and reproduce the archived human-expert and
LLM-judge agreement statistics without API access:

```bash
python -m pip install -e '.[analysis]'
python analysis/python/judge_validation.py \
  --output results/summaries/judge_validation.json
```

The script reports expert agreement, quadratic weighted Cohen's kappa for the
reasoning rubric, and absolute-agreement ICC(2,1) for stance ratings.

## Tests

```bash
python analysis/python/release_audit.py
pytest -q
```

## License

The project software is released under the MIT License. Third-party data and
model weights retain their original terms. See
[Third-Party Data and Models](THIRD_PARTY_NOTICES.md) and
[Data](data/README.md) for provenance and redistribution boundaries.

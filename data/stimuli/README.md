# Experimental Stimuli

This directory contains the fixed text inputs used by the experiment
runners. Files are UTF-8 JSON arrays whose order is part of the experimental
record.

## Value Judgment

- `value_judgment/extraction_statements.json`: 134 statements used to derive
  and orient the VAA.
- `value_judgment/held_out_statements.json`: 41 additional statements. Together
  the two files form the 175-item set used for intervention analyses.

## Sentiment Analysis

- `sentiment_analysis/development_headlines.json`: 134 news headlines.
- `sentiment_analysis/held_out_headlines.json`: 41 additional headlines.
  Together the two files form the 175-item cross-domain set.

Dataset provenance and redistribution terms will be recorded in the final data
release metadata. The filenames distinguish their role in this repository and
do not imply new train/test fitting of the language models.

## Valence Axis Extraction

- `valence/word_pairs.json`: 80 positive/negative word pairs across epistemic,
  utilitarian, deontic, and affective domains. Presenting both words separately
  produces the 160-item Valence Axis extraction set.

## Single-Letter Order

- `single_letter_order/letter_pairs.json`: 50 ordered pairs of English letters.
  Each pair is presented in both directions, producing 100 statements balanced
  between true and false. The same statements are used for the right/wrong and
  matched true/false answer-label conditions.

## Subjective Preference

- `subjective_preference/word_pairs.json`: 209 lexical pairs crossing valence
  status with lexical relation: 80 valenced opposites, 80 valenced
  non-opposites, 24 neutral contrasts, and 25 neutral non-opposites. Valenced
  non-opposites retain the positive vocabulary and pair each item with a
  negative word shifted by seven positions within the same domain. Each pair is
  presented in both display orders.

## Arithmetic Answering and Verification

- `arithmetic_answering_verification/expressions.json`: 150 unique small
  additions with operands from 2 to 49 and sums below 100. Each row includes
  the correct sum and one nearby incorrect candidate offset by 1 to 9 in either
  direction. The file materializes the deterministic stimulus generation used
  for the reported experiment (NumPy seed `20260630`) so reruns do not
  depend on regenerating random inputs.

## Alphabetical Order

- `alphabetical_order/word_pairs.json`: 30 registered word pairs, with 15
  different-initial and 15 same-initial pairs. The public runner presents both
  registered A/B orders under Think-then-Answer and Answer-then-Think prompts,
  producing 120 prompt rows before intervention levels are crossed.

## Factual Judgment

- `factual_judgment/questions.json`: the fixed 30-question Yes/No subset of
  TruthfulQA used in the manuscript. Original category, answer, source, and
  statement fields are retained alongside a normalized Boolean truth label.

## Stance-Taking

- `stance_taking/statements.json`: 30 controversial statements spanning the
  registered topic groups. English prompts, Chinese reference text, and source
  identifiers are retained in their original order.

## Preference-Induced Sycophancy

- `feedback_induced_sycophancy/argument_selection.json`: identifiers and split
  membership for 296 arguments selected from the argument-feedback data
  released with *Towards Understanding Sycophancy in Language Models*. The
  selection includes the fixed 100-argument intervention subset and its 50/50
  calibration and holdout partition.

The tracked stimulus directory omits argument text because the upstream
repository does not specify a dataset licence. Executed prompts are retained
separately in the complete archival model-output records and remain subject to
the upstream terms. See `THIRD_PARTY_NOTICES.md` and the companion code
repository's [reproduction guide](https://github.com/Yilong-Lu/Valence-Assent-Axis/blob/main/docs/reproduction.md)
for attribution and the local preparation command.

## Judge Validation

`data/judge_validation/` contains the archived validation samples and anonymous
ratings used to select the manuscript's evaluator models:

- `evaluated_results.csv`: 120 reasoning responses with two expert ratings;
- `eval_sample_sample_models_*.json`: candidate-judge scores for the same 120
  responses;
- `stance_evaluated_results.csv`: the 30-response stance validation sample;
- `answer_stance.csv` and `think_stance.csv`: two expert ratings and three
  candidate-judge ratings for final stance and reasoning stance.

These are archived validation records. Reproducing their agreement statistics
does not require API access or new model inference.

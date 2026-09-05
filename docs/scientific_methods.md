# Scientific Implementation Notes

This document records the conventions shared by the public experiment runners.
Task-specific statistical models and figure preparation are documented with
their analysis code.

## VAA construction

The VAA is derived from the 134-item extraction subset of the binary Value
Judgment task. For each statement, the assistant-start hidden state is recorded
after the selected transformer block. For each layer, the states are centered
across statements and decomposed by singular value decomposition. The first
right singular vector is PC1 and has unit Euclidean norm.

PC1 has an arbitrary sign. It is oriented so that its item scores correlate
positively with the model's probability of the support label `A`. No external
human labels enter this orientation step.

For the layer-wise binary/continuous comparison, the two PC1 directions are
estimated on non-overlapping statement subsets. Their response correlations
use format-matched scores: support probability for the binary prompt and the
probability-weighted 0--9 response for the continuous prompt. Absolute
correlations are reported because the sign of each PC1 is arbitrary.

## Intervention coefficient

Each model has a registered raw interval `(alpha_min, alpha_max)`. A normalized
coefficient in `[-1, 1]` is mapped piecewise: negative values are multiplied by
`abs(alpha_min)`, and positive values by `abs(alpha_max)`. The persistent hook
adds the resulting raw coefficient times the VAA vector at every sequence
position whenever the selected block is evaluated.

## Value and sentiment response scores

The binary Value Judgment candidates are `A` (support) and `B` (oppose). The
binary Sentiment Analysis candidates are `M` (positive) and `N` (negative).
Continuous conditions use the ten single-token candidates `0` through `9`.

The reported survey score is the unconditioned probability-weighted response:

```text
expected_response = sum(P(candidate) * candidate_value)
```

Here, candidate probabilities come from the full next-token softmax. They are
not renormalized within the candidate set. Public raw results also include the
total candidate probability mass and the candidate-conditional expectation as
format-following diagnostics; these diagnostics do not replace the reported
estimand.

## Valence and Objective Truth axes

The Valence Axis is PC1 of 160 assistant-start states elicited by 80 positive
and 80 negative words. The Single-Letter Order task presents 50 letter pairs in
both directions, yielding 100 statements balanced between true and false. Its
PC1 defines the Objective Truth Axis. The answer-label control changes only
`right/wrong` to `true/false`; the statements, ordering, extraction position,
model, and target layer remain fixed. Mistral retains its registered prompt
prefix in both label conditions.

PC1 is oriented for consistent interpretation: positive words receive the
positive Valence direction and true statements receive the positive Objective
Truth direction. Because PCA signs are arbitrary, reported comparisons use the
absolute item-level projection correlation and absolute direct axis alignment.
The raw outputs retain signed Pearson and cosine values as well as the fraction
of task-state variance captured by the unit VAA vector.

The public implementation uses a deterministic full singular value
decomposition. This avoids the run-to-run variation of approximate PCA solvers;
selected-layer metrics agree with the reported experiment outputs at manuscript
reporting precision.

For Single-Letter Order, raw results distinguish two behavioral readouts. The
candidate answer is the higher-probability registered label at the first
assistant token. The sampled response reproduces the reported one-token,
temperature-0.2 generation protocol with an explicit seed. Parse rate and
accuracy are reported separately so malformed output is not silently treated
as a valid label.

## Subjective Preference

The Subjective Preference task crosses valence status (valenced or neutral)
with lexical relation (opposite/contrast or non-opposite). Every pair is shown
in both AB and BA orders using the content-free prompt registered as
`subjective_preference_neutral_context_v1`. Candidate A retains its identity
when display order is reversed. The primary score is the complete candidate
sequence log-probability difference, A minus B, under persistent intervention
at the 11 normalized alpha levels from -1 to 1.

For canonical differences `d_AB` and `d_BA`, the order-invariant semantic and
option-order components are:

```text
semantic = (d_AB + d_BA) / 2
position = (d_AB - d_BA) / 2
```

Valenced pairs are oriented as positive minus negative. Because neutral A/B
labels have no intrinsic direction, their semantic components and initial-state
deltas are oriented as the ASCII case-insensitive alphabetically earlier word
minus the later word. This orientation does not change the position component.

The first candidate token difference is retained as a robustness field derived
from the same full-sequence scores. To keep the primary and robustness analyses
on the same model-specific item set, the tokenizer audit excludes a pair when
both candidates share their first token ID. Every excluded pair ID is recorded
in run metadata. The unsteered final input-token state is separately projected
onto the VAA for the reported initial-state AB/BA analysis.

## Arithmetic Answering and Verification

The fixed set contains 150 unique additions with operands from 2 to 49 and
results below 100. Each item has one nearby incorrect numeric candidate. The
same expression is presented as a direct numeric question, a correct equation
requiring True/False verification, and an incorrect equation requiring the same
labels. The direct condition compares the complete correct and incorrect number
sequences; both verification conditions compare the complete `True` and `False`
sequences.

Candidate accuracy is one when the correct candidate has the higher summed
sequence log probability. For verification prompts, the stored assent margin is
always `log P(True) - log P(False)`, whereas the correctness margin changes sign
for false statements. This distinction makes the directional effect on assent
auditable without changing the accuracy estimand used in the manuscript.

All three conditions use the same persistent intervention and 11 normalized
alpha levels from -1 to 1. Per-item accuracy slopes are deterministic summaries
of the binary candidate outcomes across this grid. Continuous candidate margins
are retained for diagnostics, but direct numeric results in the manuscript are
reported as accuracy.

## Open-ended reasoning and stance tasks

Alphabetical Order uses 30 registered word pairs at two difficulty levels. Each
pair is presented in both orders under Think-then-Answer and Answer-then-Think
instructions. Factual Judgment uses 30 Yes/No questions from TruthfulQA.
Stance-Taking uses 30 controversial statements and requests a critical reason
and one-sentence conclusion. The exact prompt strings used in the experiments
are versioned in `configs/prompts.json` and are not silently corrected by the
runners.

The shared decoder samples at temperature 0.2 with top-p 1.0, top-k 50, and a
512-token output limit. Persistent intervention is applied at every selected
layer invocation during generation. The original experiments did not store a
random seed; reruns use and record an explicit seed together with every other
decoding parameter. This makes each rerun traceable without implying that
stochastic completions will be text-identical to the archived manuscript data.

For the two objective tasks, the response parser reports strict JSON validity
separately from a conservative extraction of one unique quoted `answer` field.
For an item whose registered correct response is the positive assent label
(`right` or `Yes`), truth direction is +1; it is -1 for `wrong` or `No`.
Alignment Pressure is the normalized intervention coefficient multiplied by
this direction. The raw intervention applied to the model is unchanged; the
aligned quantity is an analysis coordinate that places pressure toward the
correct response on the positive side.

Stance-Taking retains the generated reasoning and conclusion fields for later
stance-related evaluation. The generation runner does not embed evaluator API
calls or evaluator outputs.

## Preference-Induced Sycophancy

The fixed stimulus pool contains 296 unique arguments. Each argument is
presented under three within-item conditions: No Preference, User Likes, and User
Dislikes. The preference suffix is appended before the quoted argument. The
model is asked for one brief reason and a final `Strong` or `Weak` verdict.

The preference-effect analysis runs all 296 arguments at zero intervention
strength. At the first model pass after the chat template opens the assistant
turn, the final prompt-position hidden state is recorded before any VAA
addition and projected onto the unit VAA. This is the assistant-start state
used to compare preference conditions. Greedy generation then continues for at
most 160 tokens. Verdict parsing accepts one unambiguous `Final verdict:
Strong/Weak` or `Verdict: Strong/Weak` expression and separately records
malformed outputs and trailing text.

The intervention analysis uses a fixed 100-argument subset and the same three
preference conditions. It applies the persistent intervention at all 11
normalized coefficients from -1 to 1 and otherwise uses the identical prompt,
decoder, activation endpoint, and verdict parser. The public output stores the
pre-addition and post-addition VAA projections, raw completion, verdict, and
parse status for every row, together with generated token IDs and their
stepwise log probabilities. State z scores use the population standard
deviation of the model's No Preference, alpha-zero assistant-start states.

No external evaluator model is used for this task; the reported behavioral
outcome is the parsed Strong-verdict indicator.

## Decoding temperature and prompt spelling

Both robustness checks use the same 60-item panel: all 30 Factual Judgment
questions and one statement from each of the 30 Alphabetical Order word pairs.
For Alphabetical Order, even-indexed pairs are shown in correct order and
odd-indexed pairs in reverse order, yielding 15 `right` and 15 `wrong` items.
Only the Think-then-Answer condition is used.

The Prompt-Spelling Check compares the original prompt with a corrected
version in which the single word `anwer` is replaced by `answer`. All other
text, punctuation, chat templating, generation settings, and response parsing
are held fixed. It uses normalized alpha values -0.2, 0, and 0.2, temperature
0.2, and seeds 0, 1, and 2.

The Decoding-Temperature Sensitivity analysis retains the original prompt and
uses normalized alpha values -0.6, -0.2, 0, 0.2, and 0.6. Temperature zero is
greedy and has one deterministic run. Temperatures 0.2 and 1 use sampling with
seeds 0, 1, and 2, top-p 1, and top-k 50. Both analyses use a 512-token output
limit and persistent VAA intervention.

Strict JSON compliance and recovery of one unambiguous quoted `answer` field
are recorded separately. Behavioral summaries use the recovered answer field;
malformed or conflicting answers remain missing. The registered model panel is
Qwen2.5-7B/14B/32B/72B and Llama-3.1-8B.

## Judge validation

The reasoning validation set contains 120 responses independently rated by two
experts for factual correctness, logical consistency, and reasoning structure.
Expert disagreements are excluded separately for each dimension when comparing
candidate judges with the expert consensus. Agreement is reported with exact
percent agreement and quadratic weighted Cohen's kappa.

The stance validation set contains 30 responses with two expert ratings and
scores from each candidate judge for final-answer stance and reasoning stance.
Inter-expert reliability and judge agreement with the mean expert rating use a
two-way random-effects, absolute-agreement, single-measure intraclass
correlation coefficient, ICC(2,1). These analyses use archived scores and make
no evaluator API calls.

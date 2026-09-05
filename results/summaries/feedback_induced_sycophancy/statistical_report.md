# Preference-Induced Sycophancy Statistical Report

## 1. Design and reporting unit

The Preference-Induced Sycophancy Task crossed each argument with three user-preference conditions: no stated preference, user liking, and user disliking. All models received the same prompt template. The alpha-zero analysis used 296 unique arguments. The intervention analysis used a fixed stratified subset of 100 arguments at 11 normalized steering strengths from -1.0 to +1.0.

Argument, not model, was the repeated-measures unit. Each of the eight models was analyzed separately; models were not treated as random replicates from a population of models. Qwen2.5-14B was the primary model, and the other models were prespecified model-specific replications. No multiplicity adjustment was applied across models or planned contrasts. All tests were two-sided.

## 2. Outcomes and statistical methods

### 2.1 Natural assistant-start VAA state

The state endpoint was the pre-addition target-layer projection at the assistant-start boundary, standardized within model by the no-preference alpha-zero distribution. Main-text inference used the two prespecified item-level paired contrasts: User Likes minus No Preference and User Dislikes minus No Preference. The user-like minus user-dislike contrast was retained as a supplementary bipolar summary. Each contrast used a two-sided paired t test and a 5,000-resample item-bootstrap 95% confidence interval.

Contrast estimates are shifts in no-preference standard-deviation units (Delta z), and paired standardized effects are reported as d_z. A supplementary omnibus test used Hotelling's T-squared on the two-dimensional paired-difference vector. This multivariate test allows an unstructured within-argument covariance matrix and does not require equal condition variances or sphericity. An unstructured repeated-measures GLS fit reproduced the paired estimates and standard errors as a numerical check.

Across all models, the largest absolute difference between an unstructured-GLS standard error and its paired-analysis counterpart was 2.758968e-07, confirming the planned paired analysis.

### 2.2 Terminal Strong/Weak verdict

The parsed verdict was binary. The three related conditions were compared with Cochran's Q, which is the paired-binary chi-square test; an ordinary Pearson chi-square test of aggregated percentages would violate independence. The Q-based effect size was W = Q/[N(k-1)]. Planned pairwise comparisons used the exact McNemar/binomial test on discordant pairs. The primary pairwise effect size was the paired risk difference in percentage points with a 5,000-resample item-bootstrap 95% confidence interval. A Haldane-corrected matched odds ratio (OR_m) is also tabulated.

### 2.3 VAA intervention curves

Normalized alpha was divided by the population SD of the fixed 11-level grid to obtain z_a. For each model, a binomial logistic mixed model estimated `verdict_strong ~ z_a * condition + (1 + z_a | item_id)`. A prespecified uncorrelated random-intercept/slope structure was used only when the correlated fit was singular or failed to converge. Simple slopes are reported as log-odds changes and odds ratios per one-SD increase in alpha. The interaction was tested by likelihood-ratio chi-square comparison.

No Strong-versus-Weak candidate-logit contrast was recorded at a shared context. Generated-token log probabilities score only the continuation that was actually produced and were not analyzed as a continuous verdict endpoint. No mediation analysis was conducted.

## 3. Results

### 3.1 Data completeness

All 296 arguments had complete VAA-state measurements in all three alpha-zero conditions for every model. Strictly parsed verdict complete cases ranged from 287 to 296 paired arguments. The intervention analysis retained 26,368 of 26,400 rows (99.88%).

| Model | State items | Verdict items | Intervention valid rows |
| --- | --- | --- | --- |
| Qwen2.5-3B | 296 | 295 | 3298/3300 |
| Qwen2.5-7B | 296 | 296 | 3300/3300 |
| Llama-3.1-8B | 296 | 296 | 3300/3300 |
| Mistral-7B | 296 | 295 | 3298/3300 |
| Gemma-2-9B | 296 | 296 | 3294/3300 |
| Qwen2.5-14B | 296 | 296 | 3300/3300 |
| Qwen2.5-32B | 296 | 296 | 3298/3300 |
| Qwen2.5-72B | 296 | 287 | 3280/3300 |

### 3.2 Stated user preferences shifted the natural assistant-start VAA state

For the primary Qwen2.5-14B model, user liking shifted the assistant-start VAA state by +0.289 [+0.263, +0.315] no-preference SD (d_z = 1.28, paired *t*(295) = 21.96, *p* < .001), whereas user disliking shifted it by -0.539 [-0.602, -0.476] no-preference SD (d_z = -0.99, paired *t*(295) = -16.99, *p* < .001).

The User Dislikes minus No Preference contrast was negative and significant in all eight models. The User Likes minus No Preference contrast was positive and significant in seven models. Qwen2.5-72B was the sole state-level exception, with a near-zero estimate and a confidence interval spanning zero.

**Table 1. Planned paired VAA-state contrasts, Delta z [item-bootstrap 95% CI]**

| Model | User Likes - No Preference | User Dislikes - No Preference | User Likes - User Dislikes |
| --- | --- | --- | --- |
| Qwen2.5-3B | +0.397 [+0.347, +0.446]; d_z = 0.93; p < .001 | -1.029 [-1.117, -0.940]; d_z = -1.31; p < .001 | +1.426 [+1.320, +1.529]; d_z = 1.56; p < .001 |
| Qwen2.5-7B | +0.110 [+0.090, +0.131]; d_z = 0.62; p < .001 | -0.581 [-0.636, -0.527]; d_z = -1.21; p < .001 | +0.691 [+0.629, +0.751]; d_z = 1.32; p < .001 |
| Llama-3.1-8B | +0.306 [+0.280, +0.332]; d_z = 1.36; p < .001 | -0.627 [-0.677, -0.579]; d_z = -1.42; p < .001 | +0.933 [+0.865, +0.998]; d_z = 1.59; p < .001 |
| Mistral-7B | +1.189 [+1.161, +1.218]; d_z = 4.83; p < .001 | -1.033 [-1.107, -0.957]; d_z = -1.54; p < .001 | +2.222 [+2.135, +2.306]; d_z = 2.98; p < .001 |
| Gemma-2-9B | +0.765 [+0.734, +0.795]; d_z = 2.79; p < .001 | -0.943 [-1.007, -0.877]; d_z = -1.63; p < .001 | +1.708 [+1.620, +1.793]; d_z = 2.19; p < .001 |
| Qwen2.5-14B | +0.289 [+0.263, +0.315]; d_z = 1.28; p < .001 | -0.539 [-0.602, -0.476]; d_z = -0.99; p < .001 | +0.828 [+0.756, +0.900]; d_z = 1.27; p < .001 |
| Qwen2.5-32B | +0.061 [+0.016, +0.105]; d_z = 0.15; p = .009 | -0.552 [-0.621, -0.482]; d_z = -0.90; p < .001 | +0.613 [+0.518, +0.708]; d_z = 0.73; p < .001 |
| Qwen2.5-72B | -0.017 [-0.079, +0.044]; d_z = -0.03; p = .590 | -2.241 [-2.347, -2.134]; d_z = -2.40; p < .001 | +2.224 [+2.070, +2.378]; d_z = 1.63; p < .001 |

**Supplementary Table 1. Omnibus repeated-measures robustness test**

| Model | Test | partial eta^2 | p |
| --- | --- | --- | --- |
| Qwen2.5-3B | T2 = 723.38; F(2, 294) = 360.47 | 0.710 | p < .001 |
| Qwen2.5-7B | T2 = 516.95; F(2, 294) = 257.60 | 0.637 | p < .001 |
| Llama-3.1-8B | T2 = 770.27; F(2, 294) = 383.83 | 0.723 | p < .001 |
| Mistral-7B | T2 = 7127.21; F(2, 294) = 3551.53 | 0.960 | p < .001 |
| Gemma-2-9B | T2 = 2308.81; F(2, 294) = 1150.49 | 0.887 | p < .001 |
| Qwen2.5-14B | T2 = 600.44; F(2, 294) = 299.20 | 0.671 | p < .001 |
| Qwen2.5-32B | T2 = 248.49; F(2, 294) = 123.82 | 0.457 | p < .001 |
| Qwen2.5-72B | T2 = 3108.53; F(2, 294) = 1549.00 | 0.913 | p < .001 |

### 3.3 Stated user preferences shifted the terminal verdict

For Qwen2.5-14B, the three preference conditions differed in Strong-verdict probability, *Q*(2) = 129.88, *p* < .001, W = 0.219. Relative to No Preference, user liking increased Strong verdicts by +9.5 [+5.4, +13.5] percentage points, exact *p* < .001; user disliking decreased them by -20.9 [-25.7, -16.2] percentage points, exact *p* < .001.

Cochran's Q was significant in all eight models, and all 24 planned paired verdict contrasts were significant in their model-specific tests. User liking increased Strong verdicts in every model; user disliking decreased them in every model.

**Table 2. Omnibus preference effect on the binary verdict**

| Model | N | Test | W | p |
| --- | --- | --- | --- | --- |
| Qwen2.5-3B | 295 | Q(2) = 364.06 | 0.617 | p < .001 |
| Qwen2.5-7B | 296 | Q(2) = 164.86 | 0.278 | p < .001 |
| Llama-3.1-8B | 296 | Q(2) = 191.39 | 0.323 | p < .001 |
| Mistral-7B | 295 | Q(2) = 367.77 | 0.623 | p < .001 |
| Gemma-2-9B | 296 | Q(2) = 175.88 | 0.297 | p < .001 |
| Qwen2.5-14B | 296 | Q(2) = 129.88 | 0.219 | p < .001 |
| Qwen2.5-32B | 296 | Q(2) = 76.88 | 0.130 | p < .001 |
| Qwen2.5-72B | 287 | Q(2) = 176.48 | 0.307 | p < .001 |

**Table 3. Planned verdict contrasts, risk difference in percentage points [95% CI]**

| Model | User Likes - No Preference | User Dislikes - No Preference | User Likes - User Dislikes |
| --- | --- | --- | --- |
| Qwen2.5-3B | +23.1 [+18.0, +28.1] pp; OR_m = 13.36; p < .001 | -56.6 [-62.4, -50.8] pp; OR_m = 0.003; p < .001 | +79.7 [+74.9, +84.1] pp; OR_m = 157.67; p < .001 |
| Qwen2.5-7B | +14.5 [+10.5, +18.9] pp; OR_m = 29.67; p < .001 | -23.0 [-28.0, -18.2] pp; OR_m = 0.02; p < .001 | +37.5 [+32.1, +43.2] pp; OR_m = 75.00; p < .001 |
| Llama-3.1-8B | +24.0 [+19.3, +29.1] pp; OR_m = 143.00; p < .001 | -18.9 [-23.6, -14.5] pp; OR_m = 0.009; p < .001 | +42.9 [+37.5, +48.6] pp; OR_m = 255.00; p < .001 |
| Mistral-7B | +24.1 [+19.3, +29.2] pp; OR_m = 48.33; p < .001 | -55.6 [-61.4, -49.8] pp; OR_m = 0.009; p < .001 | +79.7 [+74.9, +84.1] pp; OR_m = 471.00; p < .001 |
| Gemma-2-9B | +23.3 [+18.6, +28.0] pp; OR_m = 47.00; p < .001 | -16.2 [-20.3, -12.2] pp; OR_m = 0.01; p < .001 | +39.5 [+34.1, +44.9] pp; OR_m = 235.00; p < .001 |
| Qwen2.5-14B | +9.5 [+5.4, +13.5] pp; OR_m = 5.31; p < .001 | -20.9 [-25.7, -16.2] pp; OR_m = 0.04; p < .001 | +30.4 [+25.3, +35.8] pp; OR_m = 181.00; p < .001 |
| Qwen2.5-32B | +12.2 [+8.4, +16.2] pp; OR_m = 25.00; p < .001 | -6.1 [-9.5, -2.7] pp; OR_m = 0.20; p < .001 | +18.2 [+13.9, +22.6] pp; OR_m = 109.00; p < .001 |
| Qwen2.5-72B | +16.0 [+11.8, +20.6] pp; OR_m = 93.00; p < .001 | -24.4 [-29.3, -19.5] pp; OR_m = 0.007; p < .001 | +40.4 [+34.8, +46.0] pp; OR_m = 233.00; p < .001 |

### 3.4 Direct VAA intervention retained a positive behavioral orientation

For Qwen2.5-14B, all three condition-specific alpha slopes were positive and significant (all *p* < .001). The alpha-by-preference-condition interaction was also significant, likelihood-ratio chi-square(2) = 8.30, *p* = .016. Thus, stated preference changed curve location or steepness, but did not reverse the behavioral orientation of VAA.

Across all models, all 24 model-by-condition simple slopes were positive and significant in their model-specific tests. Seven of eight alpha-by-preference-condition interaction tests were significant. The primary inference is the positive simple slope in every preference condition, not equality of the three curve shapes. Odds ratios are large because several curves approach deterministic Weak/Strong responses near the grid extremes; the log-odds slope and its confidence interval should therefore be reported alongside each OR.

**Table 4. Alpha-by-preference-condition interaction and model diagnostics**

| Model | Valid rows | Random effects | Interaction | p |
| --- | --- | --- | --- | --- |
| Qwen2.5-3B | 3298 | (1 + z_a || item_id) | chi-square(2) = 143.59 | p < .001 |
| Qwen2.5-7B | 3300 | (1 + z_a | item_id) | chi-square(2) = 2.15 | p = .342 |
| Llama-3.1-8B | 3300 | (1 + z_a | item_id) | chi-square(2) = 17.42 | p < .001 |
| Mistral-7B | 3298 | (1 + z_a | item_id) | chi-square(2) = 20.19 | p < .001 |
| Gemma-2-9B | 3294 | (1 + z_a | item_id) | chi-square(2) = 76.06 | p < .001 |
| Qwen2.5-14B | 3300 | (1 + z_a | item_id) | chi-square(2) = 8.30 | p = .016 |
| Qwen2.5-32B | 3298 | (1 + z_a | item_id) | chi-square(2) = 6.46 | p = .040 |
| Qwen2.5-72B | 3280 | (1 + z_a | item_id) | chi-square(2) = 10.73 | p = .005 |

**Table 5. Condition-specific intervention slopes per one-SD alpha increase**

| Model | Condition | b [95% CI] | OR [95% CI] | p |
| --- | --- | --- | --- | --- |
| Qwen2.5-3B | baseline | +6.57 [+5.45, +7.69] | 715.42 [233.58, 2191.17] | p < .001 |
| Qwen2.5-3B | user_like | +7.59 [+6.25, +8.94] | 1988.07 [518.60, 7621.28] | p < .001 |
| Qwen2.5-3B | user_dislike | +23.58 [+18.90, +28.26] | 1.74e+10 [1.62e+08, 1.88e+12] | p < .001 |
| Qwen2.5-7B | baseline | +14.86 [+11.55, +18.16] | 2.84e+06 [1.04e+05, 7.74e+07] | p < .001 |
| Qwen2.5-7B | user_like | +14.53 [+11.11, +17.95] | 2.04e+06 [6.71e+04, 6.22e+07] | p < .001 |
| Qwen2.5-7B | user_dislike | +15.96 [+12.60, +19.32] | 8.56e+06 [2.97e+05, 2.46e+08] | p < .001 |
| Llama-3.1-8B | baseline | +6.84 [+5.68, +8.01] | 936.25 [291.81, 3003.94] | p < .001 |
| Llama-3.1-8B | user_like | +7.51 [+6.13, +8.89] | 1829.09 [459.84, 7275.53] | p < .001 |
| Llama-3.1-8B | user_dislike | +9.39 [+7.69, +11.08] | 1.20e+04 [2195.67, 6.51e+04] | p < .001 |
| Mistral-7B | baseline | +4.02 [+3.39, +4.65] | 55.71 [29.54, 105.06] | p < .001 |
| Mistral-7B | user_like | +3.52 [+2.85, +4.19] | 33.80 [17.30, 66.07] | p < .001 |
| Mistral-7B | user_dislike | +4.90 [+4.23, +5.57] | 134.24 [68.50, 263.06] | p < .001 |
| Gemma-2-9B | baseline | +12.35 [+9.62, +15.07] | 2.30e+05 [1.51e+04, 3.50e+06] | p < .001 |
| Gemma-2-9B | user_like | +11.78 [+8.86, +14.70] | 1.31e+05 [7052.75, 2.43e+06] | p < .001 |
| Gemma-2-9B | user_dislike | +22.49 [+17.61, +27.37] | 5.84e+09 [4.44e+07, 7.67e+11] | p < .001 |
| Qwen2.5-14B | baseline | +20.21 [+14.84, +25.59] | 6.01e+08 [2.77e+06, 1.30e+11] | p < .001 |
| Qwen2.5-14B | user_like | +19.84 [+14.33, +25.35] | 4.13e+08 [1.68e+06, 1.02e+11] | p < .001 |
| Qwen2.5-14B | user_dislike | +23.35 [+17.49, +29.21] | 1.38e+10 [3.93e+07, 4.87e+12] | p < .001 |
| Qwen2.5-32B | baseline | +22.08 [+15.13, +29.03] | 3.88e+09 [3.72e+06, 4.04e+12] | p < .001 |
| Qwen2.5-32B | user_like | +22.85 [+15.35, +30.35] | 8.37e+09 [4.63e+06, 1.51e+13] | p < .001 |
| Qwen2.5-32B | user_dislike | +21.03 [+14.39, +27.67] | 1.36e+09 [1.78e+06, 1.04e+12] | p < .001 |
| Qwen2.5-72B | baseline | +11.64 [+9.13, +14.16] | 1.14e+05 [9191.14, 1.41e+06] | p < .001 |
| Qwen2.5-72B | user_like | +12.91 [+10.00, +15.83] | 4.06e+05 [2.20e+04, 7.52e+06] | p < .001 |
| Qwen2.5-72B | user_dislike | +11.20 [+8.91, +13.48] | 7.28e+04 [7428.01, 7.13e+05] | p < .001 |

**Table 6. Differences between condition-specific alpha slopes**

| Model | User Likes - No Preference | User Dislikes - No Preference | User Likes - User Dislikes |
| --- | --- | --- | --- |
| Qwen2.5-3B | +1.02 [+0.06, +1.98]; p = .037 | +17.01 [+12.73, +21.29]; p < .001 | -15.99 [-20.25, -11.73]; p < .001 |
| Qwen2.5-7B | -0.33 [-1.74, +1.09]; p = .650 | +1.10 [-0.77, +2.97]; p = .247 | -1.43 [-3.41, +0.55]; p = .156 |
| Llama-3.1-8B | +0.67 [-0.06, +1.40]; p = .071 | +2.55 [+1.17, +3.93]; p < .001 | -1.88 [-3.37, -0.39]; p = .013 |
| Mistral-7B | -0.50 [-1.07, +0.07]; p = .086 | +0.88 [+0.32, +1.44]; p = .002 | -1.38 [-2.01, -0.74]; p < .001 |
| Gemma-2-9B | -0.56 [-1.39, +0.27]; p = .183 | +10.14 [+6.31, +13.98]; p < .001 | -10.71 [-14.61, -6.80]; p < .001 |
| Qwen2.5-14B | -0.37 [-1.99, +1.24]; p = .649 | +3.14 [+0.54, +5.73]; p = .018 | -3.51 [-6.18, -0.84]; p = .010 |
| Qwen2.5-32B | +0.77 [-0.62, +2.16]; p = .278 | -1.05 [-2.24, +0.14]; p = .084 | +1.82 [+0.26, +3.37]; p = .022 |
| Qwen2.5-72B | +1.27 [+0.35, +2.19]; p = .007 | -0.45 [-1.30, +0.41]; p = .307 | +1.72 [+0.57, +2.87]; p = .003 |

## 4. Statistical conclusion and claim boundary

Stated user preferences shifted both the natural assistant-start VAA state and the later Strong/Weak evaluation. Direct intervention along the VAA increased the probability of a Strong verdict under every preference condition in every model. Together, these findings show that preference-induced sycophancy is reflected in the assistant-start VAA state while the behavioral orientation of VAA intervention is preserved.

These analyses do not test statistical mediation or assume that a stated user preference is equivalent to a fixed intervention magnitude. Generated reasons were retained for audit but were not an analysis endpoint.

## 5. Reproducibility

Canonical script: `analysis/r/feedback_induced_sycophancy.R`.

Machine-readable estimates, unadjusted model-specific p values, diagnostics, analysis metadata, and R session information are stored beside this report.

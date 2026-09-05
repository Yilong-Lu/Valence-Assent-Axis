#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(lme4)
  library(nlme)
  library(emmeans)
  library(jsonlite)
})

options(contrasts = c("contr.treatment", "contr.poly"))

script_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
if (length(script_arg) != 1L) stop("Could not resolve script path")
script_path <- normalizePath(sub("^--file=", "", script_arg))
repository_root <- normalizePath(file.path(dirname(script_path), "..", ".."))

args <- commandArgs(trailingOnly = TRUE)
input_arg <- grep("^--input-dir=", args, value = TRUE)
output_arg <- grep("^--output-dir=", args, value = TRUE)
input_dir <- if (length(input_arg)) {
  normalizePath(sub("^--input-dir=", "", input_arg[[1]]))
} else {
  file.path(repository_root, "data", "processed", "feedback_induced_sycophancy")
}
output_dir <- if (length(output_arg)) {
  normalizePath(sub("^--output-dir=", "", output_arg[[1]]), mustWork = FALSE)
} else {
  file.path(repository_root, "results", "summaries", "feedback_induced_sycophancy")
}
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

models <- c(
  "qwen25_3b", "qwen25_7b", "llama3_8b", "mistral_7b",
  "gemma2_9b", "qwen25_14b", "qwen25_32b", "qwen25_72b"
)
condition_levels <- c("baseline", "user_like", "user_dislike")
alpha_grid <- seq(-1, 1, by = 0.2)
alpha_scale <- sqrt(mean(alpha_grid^2))
state_metric <- "pre_addition_vaa_projection_unit_z_baseline"

contrast_list <- list(
  user_like_minus_baseline = c(-1, 1, 0),
  user_dislike_minus_baseline = c(-1, 0, 1),
  user_like_minus_user_dislike = c(0, 1, -1)
)

feedback_path <- file.path(input_dir, "feedback_effect.csv")
intervention_path <- file.path(input_dir, "intervention.csv")
if (!file.exists(feedback_path)) stop("Missing feedback-effect input: ", feedback_path)
if (!file.exists(intervention_path)) stop("Missing intervention input: ", intervention_path)
feedback_all <- read.csv(feedback_path, stringsAsFactors = FALSE, check.names = FALSE)
intervention_all <- read.csv(intervention_path, stringsAsFactors = FALSE, check.names = FALSE)

as_bool <- function(x) {
  if (is.logical(x)) return(x)
  tolower(as.character(x)) %in% c("true", "t", "1")
}

fit_with_warnings <- function(expr) {
  captured <- character()
  fit <- withCallingHandlers(
    expr,
    warning = function(w) {
      captured <<- c(captured, conditionMessage(w))
      invokeRestart("muffleWarning")
    }
  )
  list(fit = fit, warnings = unique(captured))
}

conv_messages <- function(fit) {
  messages <- fit@optinfo$conv$lme4$messages
  if (is.null(messages)) "" else paste(messages, collapse = " | ")
}

complete_items <- function(data, required_n) {
  counts <- table(data$item_id)
  names(counts[counts == required_n])
}

write_table <- function(data, filename) {
  write.csv(data, file.path(output_dir, filename), row.names = FALSE, na = "")
}

ci_column <- function(data, side) {
  candidates <- if (side == "lower") c("lower.CL", "asymp.LCL") else c("upper.CL", "asymp.UCL")
  found <- intersect(candidates, names(data))
  if (!length(found)) stop("Could not find ", side, " confidence-limit column")
  data[[found[[1]]]]
}

paired_state_tests <- function(data, model, seed, n_bootstrap = 5000L) {
  wide <- reshape(
    data[c("item_id", "condition", "vaa_state")],
    idvar = "item_id", timevar = "condition", direction = "wide"
  )
  values <- as.matrix(wide[paste0("vaa_state.", condition_levels)])
  difference_matrix <- cbind(
    user_like_minus_baseline = values[, "vaa_state.user_like"] -
      values[, "vaa_state.baseline"],
    user_dislike_minus_baseline = values[, "vaa_state.user_dislike"] -
      values[, "vaa_state.baseline"]
  )

  n_items <- nrow(difference_matrix)
  n_dimensions <- ncol(difference_matrix)
  mean_vector <- colMeans(difference_matrix)
  covariance <- cov(difference_matrix)
  hotelling_t2 <- as.numeric(
    n_items * crossprod(mean_vector, solve(covariance, mean_vector))
  )
  f_value <- (n_items - n_dimensions) * hotelling_t2 /
    (n_dimensions * (n_items - 1))
  omnibus <- data.frame(
    model_name = model,
    test = "Hotelling_T2_paired_repeated_measures",
    n_items = n_items,
    hotelling_t2 = hotelling_t2,
    numerator_df = n_dimensions,
    denominator_df = n_items - n_dimensions,
    f_value = f_value,
    partial_eta_squared =
      n_dimensions * f_value /
      (n_dimensions * f_value + n_items - n_dimensions),
    p_value = pf(
      f_value, df1 = n_dimensions, df2 = n_items - n_dimensions,
      lower.tail = FALSE
    ),
    stringsAsFactors = FALSE
  )

  specifications <- list(
    user_like_minus_baseline = values[, "vaa_state.user_like"] -
      values[, "vaa_state.baseline"],
    user_dislike_minus_baseline = values[, "vaa_state.user_dislike"] -
      values[, "vaa_state.baseline"],
    user_like_minus_user_dislike = values[, "vaa_state.user_like"] -
      values[, "vaa_state.user_dislike"]
  )
  set.seed(seed)
  contrasts <- lapply(names(specifications), function(name) {
    difference <- as.numeric(specifications[[name]])
    test <- t.test(difference, mu = 0)
    draws <- replicate(
      n_bootstrap,
      mean(sample(difference, length(difference), replace = TRUE))
    )
    paired_sd <- sd(difference)
    data.frame(
      model_name = model,
      contrast = name,
      n_pairs = length(difference),
      estimate = mean(difference),
      paired_sd = paired_sd,
      standardized_paired_effect_dz = mean(difference) / paired_sd,
      se = paired_sd / sqrt(length(difference)),
      df = unname(test$parameter),
      ci_low = unname(quantile(draws, 0.025)),
      ci_high = unname(quantile(draws, 0.975)),
      t_ratio = unname(test$statistic),
      p_value = test$p.value,
      stringsAsFactors = FALSE
    )
  })

  list(omnibus = omnibus, contrasts = do.call(rbind, contrasts))
}

state_covariance_diagnostics <- function(data, model, paired_contrasts) {
  ordered <- data[order(data$item_id, data$condition), ]
  ordered$condition_index <- as.integer(ordered$condition)
  unstructured <- gls(
    vaa_state ~ condition,
    correlation = corSymm(form = ~condition_index | item_id),
    weights = varIdent(form = ~1 | condition),
    data = ordered,
    method = "REML"
  )
  compound_symmetry <- lmer(
    vaa_state ~ condition + (1 | item_id), data = ordered, REML = TRUE
  )

  coefficient_contrasts <- rbind(
    user_like_minus_baseline = c(0, 1, 0),
    user_dislike_minus_baseline = c(0, 0, 1),
    user_like_minus_user_dislike = c(0, 1, -1)
  )
  gls_se <- sqrt(diag(
    coefficient_contrasts %*% vcov(unstructured) %*% t(coefficient_contrasts)
  ))
  lmm_se <- sqrt(diag(
    coefficient_contrasts %*% as.matrix(vcov(compound_symmetry)) %*%
      t(coefficient_contrasts)
  ))
  paired_se <- setNames(paired_contrasts$se, paired_contrasts$contrast)

  wide <- reshape(
    ordered[c("item_id", "condition", "vaa_state")],
    idvar = "item_id", timevar = "condition", direction = "wide"
  )
  values <- as.matrix(wide[paste0("vaa_state.", condition_levels)])
  condition_sds <- apply(values, 2, sd)
  condition_correlations <- cor(values)

  data.frame(
    model_name = model,
    baseline_sd = condition_sds[[1]],
    user_like_sd = condition_sds[[2]],
    user_dislike_sd = condition_sds[[3]],
    cor_baseline_like = condition_correlations[1, 2],
    cor_baseline_dislike = condition_correlations[1, 3],
    cor_like_dislike = condition_correlations[2, 3],
    equal_lmm_like_se = lmm_se[["user_like_minus_baseline"]],
    unstructured_gls_like_se = gls_se[["user_like_minus_baseline"]],
    paired_like_se = paired_se[["user_like_minus_baseline"]],
    max_abs_gls_minus_paired_se = max(abs(gls_se - paired_se[names(gls_se)])),
    equal_lmm_aic = AIC(compound_symmetry),
    unstructured_gls_aic = AIC(unstructured),
    stringsAsFactors = FALSE
  )
}

extract_glmer_omnibus <- function(full, reduced, model, test_name) {
  tab <- suppressMessages(anova(reduced, full, test = "Chisq"))
  data.frame(
    model_name = model,
    test = test_name,
    chi_square = tab$Chisq[[2]],
    df = tab$Df[[2]],
    p_value = tab$`Pr(>Chisq)`[[2]],
    stringsAsFactors = FALSE
  )
}

cochran_q_test <- function(data, model) {
  response <- xtabs(verdict_strong ~ item_id + condition, data = data)
  k <- ncol(response)
  column_sums <- colSums(response)
  row_sums <- rowSums(response)
  total <- sum(response)
  denominator <- k * total - sum(row_sums^2)
  q_value <- if (denominator == 0) 0 else {
    (k - 1) * (k * sum(column_sums^2) - total^2) / denominator
  }
  data.frame(
    model_name = model,
    test = "Cochran_Q",
    n_items = nrow(response),
    chi_square = q_value,
    df = k - 1,
    q_effect_w = q_value / (nrow(response) * (k - 1)),
    p_value = pchisq(q_value, df = k - 1, lower.tail = FALSE),
    stringsAsFactors = FALSE
  )
}

paired_verdict_contrasts <- function(data, model, seed, n_bootstrap = 5000L) {
  specifications <- list(
    user_like_minus_baseline = c("user_like", "baseline"),
    user_dislike_minus_baseline = c("user_dislike", "baseline"),
    user_like_minus_user_dislike = c("user_like", "user_dislike")
  )
  set.seed(seed)
  rows <- lapply(names(specifications), function(name) {
    levels <- specifications[[name]]
    first <- data[data$condition == levels[[1]], c("item_id", "verdict_strong")]
    second <- data[data$condition == levels[[2]], c("item_id", "verdict_strong")]
    names(first)[[2]] <- "first"
    names(second)[[2]] <- "second"
    paired <- merge(first, second, by = "item_id", sort = FALSE)
    difference <- paired$first - paired$second
    draws <- replicate(
      n_bootstrap,
      mean(sample(difference, length(difference), replace = TRUE))
    )
    positive_discordant <- sum(paired$first == 1 & paired$second == 0)
    negative_discordant <- sum(paired$first == 0 & paired$second == 1)
    n_discordant <- positive_discordant + negative_discordant
    exact_p <- if (n_discordant == 0) 1 else {
      binom.test(positive_discordant, n_discordant, p = 0.5)$p.value
    }
    data.frame(
      model_name = model,
      contrast = name,
      n_pairs = nrow(paired),
      probability_difference = mean(difference),
      probability_ci_low = unname(quantile(draws, 0.025)),
      probability_ci_high = unname(quantile(draws, 0.975)),
      positive_discordant = positive_discordant,
      negative_discordant = negative_discordant,
      matched_odds_ratio_haldane =
        (positive_discordant + 0.5) / (negative_discordant + 0.5),
      p_value = exact_p,
      stringsAsFactors = FALSE
    )
  })
  do.call(rbind, rows)
}

input_counts <- list()
feedback_state_omnibus <- list()
feedback_state_contrasts <- list()
feedback_state_diagnostics <- list()
feedback_verdict_omnibus <- list()
feedback_verdict_contrasts <- list()
intervention_diagnostics <- list()
intervention_omnibus <- list()
intervention_condition_slopes <- list()
intervention_slope_contrasts <- list()

control <- glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 200000))

for (model_index in seq_along(models)) {
  model <- models[[model_index]]
  message("Feedback effect: ", model)

  p1 <- feedback_all[feedback_all$model_name == model, ]
  p1$condition <- factor(p1$condition, levels = condition_levels)
  p1$item_id <- factor(p1$item_id)
  p1$vaa_state <- as.numeric(p1[[state_metric]])
  p1$verdict_valid <- as_bool(p1$verdict_valid)
  p1$verdict_strong <- as.numeric(p1$verdict_strong)

  if (nrow(p1) != 296L * 3L) stop(model, ": Feedback-effect analysis row count mismatch")
  if (any(abs(p1$alpha_norm) > 1e-12)) stop(model, ": Feedback-effect analysis contains nonzero alpha")
  if (!setequal(as.character(unique(p1$condition)), condition_levels)) {
    stop(model, ": Feedback-effect analysis condition mismatch")
  }
  if (anyDuplicated(p1[c("item_id", "condition")])) stop(model, ": duplicate Feedback-effect analysis rows")

  state_data <- p1[is.finite(p1$vaa_state), ]
  state_items <- complete_items(state_data, 3L)
  state_data <- state_data[state_data$item_id %in% state_items, ]
  state_data$item_id <- droplevels(state_data$item_id)

  state_tests <- paired_state_tests(
    state_data, model, seed = 20261811L + model_index
  )
  feedback_state_omnibus[[model]] <- state_tests$omnibus
  feedback_state_contrasts[[model]] <- state_tests$contrasts
  feedback_state_diagnostics[[model]] <- state_covariance_diagnostics(
    state_data, model, state_tests$contrasts
  )

  verdict_data <- p1[p1$verdict_valid & !is.na(p1$verdict_strong), ]
  verdict_items <- complete_items(verdict_data, 3L)
  verdict_data <- verdict_data[verdict_data$item_id %in% verdict_items, ]
  verdict_data$item_id <- droplevels(verdict_data$item_id)

  feedback_verdict_omnibus[[model]] <- cochran_q_test(verdict_data, model)
  feedback_verdict_contrasts[[model]] <- paired_verdict_contrasts(
    verdict_data, model, seed = 20260811L + model_index
  )

  p2 <- intervention_all[intervention_all$model_name == model, ]
  p2$condition <- factor(p2$condition, levels = condition_levels)
  p2$item_id <- factor(p2$item_id)
  p2$verdict_valid <- as_bool(p2$verdict_valid)
  p2$verdict_strong <- as.numeric(p2$verdict_strong)
  p2$z_a <- as.numeric(p2$alpha_norm) / alpha_scale

  if (nrow(p2) != 100L * 3L * 11L) stop(model, ": Intervention analysis row count mismatch")
  if (!isTRUE(all.equal(sort(unique(round(p2$alpha_norm, 10))), alpha_grid))) {
    stop(model, ": Intervention analysis alpha grid mismatch")
  }
  if (anyDuplicated(p2[c("item_id", "condition", "alpha_norm")])) {
    stop(model, ": duplicate Intervention analysis rows")
  }
  intervention_data <- p2[p2$verdict_valid & !is.na(p2$verdict_strong), ]
  if (nrow(intervention_data) != nrow(p2)) {
    message(
      model, ": Intervention analysis strict complete cases ", nrow(intervention_data),
      "/", nrow(p2)
    )
  }

  fit_intervention <- function(uncorrelated = FALSE) {
    random <- if (uncorrelated) "(1 + z_a || item_id)" else "(1 + z_a | item_id)"
    full_formula <- as.formula(paste("verdict_strong ~ z_a * condition +", random))
    reduced_formula <- as.formula(paste("verdict_strong ~ z_a + condition +", random))
    full <- fit_with_warnings(glmer(
      full_formula, data = intervention_data, family = binomial,
      control = control, nAGQ = 1
    ))
    reduced <- fit_with_warnings(glmer(
      reduced_formula, data = intervention_data, family = binomial,
      control = control, nAGQ = 1
    ))
    list(full = full, reduced = reduced, random = random)
  }

  intervention_fit <- fit_intervention(FALSE)
  initial_problem <- isSingular(intervention_fit$full$fit, tol = 1e-4) ||
    nzchar(conv_messages(intervention_fit$full$fit))
  if (initial_problem) intervention_fit <- fit_intervention(TRUE)
  full <- intervention_fit$full$fit
  reduced <- intervention_fit$reduced$fit

  intervention_diagnostics[[model]] <- data.frame(
    model_name = model,
    n_rows = nrow(intervention_data),
    n_items = length(unique(intervention_data$item_id)),
    random_effect_structure = intervention_fit$random,
    used_uncorrelated_fallback = initial_problem,
    singular = isSingular(full, tol = 1e-4),
    convergence_message = conv_messages(full),
    warnings = paste(intervention_fit$full$warnings, collapse = " | "),
    stringsAsFactors = FALSE
  )
  intervention_omnibus[[model]] <- extract_glmer_omnibus(
    full, reduced, model, "z_a_by_condition"
  )

  trends <- emtrends(full, ~ condition, var = "z_a")
  trend_summary <- as.data.frame(summary(trends, infer = c(TRUE, TRUE)))
  intervention_condition_slopes[[model]] <- data.frame(
    model_name = model,
    condition = trend_summary$condition,
    log_odds_slope_per_alpha_sd = trend_summary$z_a.trend,
    se = trend_summary$SE,
    ci_low = ci_column(trend_summary, "lower"),
    ci_high = ci_column(trend_summary, "upper"),
    odds_ratio_per_alpha_sd = exp(trend_summary$z_a.trend),
    odds_ratio_ci_low = exp(ci_column(trend_summary, "lower")),
    odds_ratio_ci_high = exp(ci_column(trend_summary, "upper")),
    z_ratio = trend_summary$z.ratio,
    p_value = trend_summary$p.value,
    stringsAsFactors = FALSE
  )
  slope_diff <- as.data.frame(summary(
    contrast(trends, method = contrast_list, adjust = "none"),
    infer = c(TRUE, TRUE)
  ))
  intervention_slope_contrasts[[model]] <- data.frame(
    model_name = model,
    contrast = slope_diff$contrast,
    log_odds_slope_difference = slope_diff$estimate,
    se = slope_diff$SE,
    ci_low = ci_column(slope_diff, "lower"),
    ci_high = ci_column(slope_diff, "upper"),
    z_ratio = slope_diff$z.ratio,
    p_value = slope_diff$p.value,
    stringsAsFactors = FALSE
  )

  input_counts[[model]] <- data.frame(
    model_name = model,
    feedback_rows = nrow(p1),
    feedback_state_complete_items = length(state_items),
    feedback_verdict_complete_items = length(verdict_items),
    intervention_rows = nrow(p2),
    intervention_strict_valid_rows = nrow(intervention_data),
    intervention_items = length(unique(p2$item_id)),
    stringsAsFactors = FALSE
  )
}

bind_rows <- function(x) do.call(rbind, unname(x))

input_counts <- bind_rows(input_counts)
feedback_state_omnibus <- bind_rows(feedback_state_omnibus)
feedback_state_contrasts <- bind_rows(feedback_state_contrasts)
feedback_state_diagnostics <- bind_rows(feedback_state_diagnostics)
feedback_verdict_omnibus <- bind_rows(feedback_verdict_omnibus)
feedback_verdict_contrasts <- bind_rows(feedback_verdict_contrasts)
intervention_diagnostics <- bind_rows(intervention_diagnostics)
intervention_omnibus <- bind_rows(intervention_omnibus)
intervention_condition_slopes <- bind_rows(intervention_condition_slopes)
intervention_slope_contrasts <- bind_rows(intervention_slope_contrasts)

write_table(input_counts, "input_counts.csv")
write_table(feedback_state_omnibus, "feedback_state_omnibus.csv")
write_table(feedback_state_contrasts, "feedback_state_contrasts.csv")
write_table(feedback_state_diagnostics, "feedback_state_covariance_diagnostics.csv")
write_table(feedback_verdict_omnibus, "feedback_verdict_omnibus.csv")
write_table(feedback_verdict_contrasts, "feedback_verdict_contrasts.csv")
write_table(intervention_diagnostics, "intervention_glmer_diagnostics.csv")
write_table(intervention_omnibus, "intervention_interaction_tests.csv")
write_table(intervention_condition_slopes, "intervention_condition_slopes.csv")
write_table(intervention_slope_contrasts, "intervention_slope_contrasts.csv")

metadata <- list(
  analysis_date = as.character(Sys.Date()),
  models = models,
  feedback_state_primary = paste(
    "Planned item-level paired t tests with paired dz and item-bootstrap",
    "confidence intervals; this is the main-text analysis"
  ),
  feedback_state_omnibus_robustness = paste(
    "Supplementary Hotelling T2 omnibus test on the paired difference vector"
  ),
  feedback_state_robustness = paste(
    "Unstructured repeated-measures GLS; condition-specific variances and all",
    "within-item correlations estimated freely"
  ),
  feedback_verdict_test = "Cochran Q omnibus plus exact paired McNemar/binomial contrasts",
  intervention_formula = "verdict_strong ~ z_a * condition + (1 + z_a | item_id)",
  intervention_fallback = "verdict_strong ~ z_a * condition + (1 + z_a || item_id)",
  alpha_grid = alpha_grid,
  alpha_population_sd = alpha_scale,
  state_metric = state_metric,
  p_value_reporting = paste(
    "Unadjusted two-sided p values for prespecified model-specific tests;",
    "Qwen2.5-14B is the primary model and the other models are independent replications"
  ),
  feedback_verdict_score_note = paste(
    "No Strong-vs-Weak candidate-logit contrast was recorded; generated-token",
    "log probabilities are not used as a continuous verdict endpoint"
  )
)
write_json(metadata, file.path(output_dir, "analysis_metadata.json"), pretty = TRUE, auto_unbox = TRUE)
session_info <- sub("[[:space:]]+$", "", capture.output(sessionInfo()))
conda_prefix <- Sys.getenv("CONDA_PREFIX", unset = "")
if (nzchar(conda_prefix)) {
  session_info <- gsub(
    normalizePath(conda_prefix), "$CONDA_PREFIX", session_info, fixed = TRUE
  )
}
writeLines(session_info, file.path(output_dir, "r_session_info.txt"))

model_labels <- c(
  qwen25_3b = "Qwen2.5-3B", qwen25_7b = "Qwen2.5-7B",
  llama3_8b = "Llama-3.1-8B", mistral_7b = "Mistral-7B",
  gemma2_9b = "Gemma-2-9B", qwen25_14b = "Qwen2.5-14B",
  qwen25_32b = "Qwen2.5-32B", qwen25_72b = "Qwen2.5-72B"
)

fmt_p <- function(p) {
  if (!is.finite(p)) return("NA")
  if (p < 0.001) return("< .001")
  paste0("= ", sub("^0", "", sprintf("%.3f", p)))
}

fmt_num <- function(x, digits = 2) sprintf(paste0("%.", digits, "f"), x)

fmt_or <- function(x) {
  if (!is.finite(x)) return("NA")
  if (x >= 1e4) return(sprintf("%.2e", x))
  if (x < 0.01) return(sprintf("%.3f", x))
  sprintf("%.2f", x)
}

fmt_est_ci <- function(est, low, high, digits = 3) {
  paste0(
    sprintf(paste0("%+.", digits, "f"), est), " [",
    sprintf(paste0("%+.", digits, "f"), low), ", ",
    sprintf(paste0("%+.", digits, "f"), high), "]"
  )
}

fmt_pct_ci <- function(est, low, high) {
  paste0(
    sprintf("%+.1f", 100 * est), " [",
    sprintf("%+.1f", 100 * low), ", ",
    sprintf("%+.1f", 100 * high), "]"
  )
}

md_table <- function(headers, rows) {
  c(
    paste0("| ", paste(headers, collapse = " | "), " |"),
    paste0("| ", paste(rep("---", length(headers)), collapse = " | "), " |"),
    apply(rows, 1, function(row) paste0("| ", paste(row, collapse = " | "), " |"))
  )
}

contrast_row <- function(data, model, contrast) {
  data[data$model_name == model & data$contrast == contrast, , drop = FALSE]
}

sample_rows <- t(vapply(seq_len(nrow(input_counts)), function(i) {
  row <- input_counts[i, ]
  c(
    model_labels[[row$model_name]], row$feedback_state_complete_items,
    row$feedback_verdict_complete_items,
    paste0(row$intervention_strict_valid_rows, "/", row$intervention_rows)
  )
}, character(4)))

state_omnibus_rows <- t(vapply(seq_len(nrow(feedback_state_omnibus)), function(i) {
  row <- feedback_state_omnibus[i, ]
  c(
    model_labels[[row$model_name]],
    paste0(
      "T2 = ", fmt_num(row$hotelling_t2, 2), "; F(2, ",
      round(row$denominator_df), ") = ", fmt_num(row$f_value, 2)
    ),
    fmt_num(row$partial_eta_squared, 3),
    paste0("p ", fmt_p(row$p_value))
  )
}, character(4)))

state_contrast_rows <- t(vapply(models, function(model) {
  like <- contrast_row(feedback_state_contrasts, model, "user_like_minus_baseline")
  dislike <- contrast_row(feedback_state_contrasts, model, "user_dislike_minus_baseline")
  bipolar <- contrast_row(feedback_state_contrasts, model, "user_like_minus_user_dislike")
  c(
    model_labels[[model]],
    paste0(
      fmt_est_ci(like$estimate, like$ci_low, like$ci_high),
      "; d_z = ", fmt_num(like$standardized_paired_effect_dz, 2),
      "; p ", fmt_p(like$p_value)
    ),
    paste0(
      fmt_est_ci(dislike$estimate, dislike$ci_low, dislike$ci_high),
      "; d_z = ", fmt_num(dislike$standardized_paired_effect_dz, 2),
      "; p ", fmt_p(dislike$p_value)
    ),
    paste0(
      fmt_est_ci(bipolar$estimate, bipolar$ci_low, bipolar$ci_high),
      "; d_z = ", fmt_num(bipolar$standardized_paired_effect_dz, 2),
      "; p ", fmt_p(bipolar$p_value)
    )
  )
}, character(4)))

verdict_omnibus_rows <- t(vapply(seq_len(nrow(feedback_verdict_omnibus)), function(i) {
  row <- feedback_verdict_omnibus[i, ]
  c(
    model_labels[[row$model_name]], row$n_items,
    paste0("Q(2) = ", fmt_num(row$chi_square, 2)),
    fmt_num(row$q_effect_w, 3),
    paste0("p ", fmt_p(row$p_value))
  )
}, character(5)))

verdict_contrast_rows <- t(vapply(models, function(model) {
  like <- contrast_row(feedback_verdict_contrasts, model, "user_like_minus_baseline")
  dislike <- contrast_row(feedback_verdict_contrasts, model, "user_dislike_minus_baseline")
  bipolar <- contrast_row(feedback_verdict_contrasts, model, "user_like_minus_user_dislike")
  cell <- function(row) paste0(
    fmt_pct_ci(row$probability_difference, row$probability_ci_low, row$probability_ci_high),
    " pp; OR_m = ", fmt_or(row$matched_odds_ratio_haldane),
    "; p ", fmt_p(row$p_value)
  )
  c(model_labels[[model]], cell(like), cell(dislike), cell(bipolar))
}, character(4)))

intervention_interaction_rows <- t(vapply(seq_len(nrow(intervention_omnibus)), function(i) {
  row <- intervention_omnibus[i, ]
  diagnostics <- intervention_diagnostics[intervention_diagnostics$model_name == row$model_name, ]
  c(
    model_labels[[row$model_name]], diagnostics$n_rows,
    diagnostics$random_effect_structure,
    paste0("chi-square(2) = ", fmt_num(row$chi_square, 2)),
    paste0("p ", fmt_p(row$p_value))
  )
}, character(5)))

intervention_slope_rows <- t(vapply(seq_len(nrow(intervention_condition_slopes)), function(i) {
  row <- intervention_condition_slopes[i, ]
  c(
    model_labels[[row$model_name]], as.character(row$condition),
    fmt_est_ci(
      row$log_odds_slope_per_alpha_sd, row$ci_low, row$ci_high, digits = 2
    ),
    paste0(
      fmt_or(row$odds_ratio_per_alpha_sd), " [",
      fmt_or(row$odds_ratio_ci_low), ", ", fmt_or(row$odds_ratio_ci_high), "]"
    ),
    paste0("p ", fmt_p(row$p_value))
  )
}, character(5)))

intervention_contrast_rows <- t(vapply(models, function(model) {
  like <- contrast_row(intervention_slope_contrasts, model, "user_like_minus_baseline")
  dislike <- contrast_row(intervention_slope_contrasts, model, "user_dislike_minus_baseline")
  bipolar <- contrast_row(intervention_slope_contrasts, model, "user_like_minus_user_dislike")
  cell <- function(row) paste0(
    fmt_est_ci(row$log_odds_slope_difference, row$ci_low, row$ci_high, digits = 2),
    "; p ", fmt_p(row$p_value)
  )
  c(model_labels[[model]], cell(like), cell(dislike), cell(bipolar))
}, character(4)))

q14_state <- feedback_state_omnibus[feedback_state_omnibus$model_name == "qwen25_14b", ]
q14_like_state <- contrast_row(feedback_state_contrasts, "qwen25_14b", "user_like_minus_baseline")
q14_dislike_state <- contrast_row(feedback_state_contrasts, "qwen25_14b", "user_dislike_minus_baseline")
q14_verdict <- feedback_verdict_omnibus[feedback_verdict_omnibus$model_name == "qwen25_14b", ]
q14_like_verdict <- contrast_row(feedback_verdict_contrasts, "qwen25_14b", "user_like_minus_baseline")
q14_dislike_verdict <- contrast_row(feedback_verdict_contrasts, "qwen25_14b", "user_dislike_minus_baseline")
q14_interaction <- intervention_omnibus[intervention_omnibus$model_name == "qwen25_14b", ]

report <- c(
  "# Preference-Induced Sycophancy Statistical Report",
  "",
  "## 1. Design and reporting unit",
  "",
  paste(
    "The Preference-Induced Sycophancy Task crossed each argument with three",
    "user-preference conditions: no stated preference, user liking, and user",
    "disliking. All models received the same prompt template.",
    "The alpha-zero analysis used 296 unique arguments. The intervention analysis",
    "used a fixed stratified subset of 100 arguments at 11 normalized",
    "steering strengths from -1.0 to +1.0."
  ),
  "",
  paste(
    "Argument, not model, was the repeated-measures unit. Each of the eight models",
    "was analyzed separately; models were not treated as random replicates from a",
    "population of models. Qwen2.5-14B was the primary model, and the other models",
    "were prespecified model-specific replications. No multiplicity adjustment was",
    "applied across models or planned contrasts. All tests were two-sided."
  ),
  "",
  "## 2. Outcomes and statistical methods",
  "",
  "### 2.1 Natural assistant-start VAA state",
  "",
  paste(
    "The state endpoint was the pre-addition target-layer projection at the",
    "assistant-start boundary, standardized within model by the no-preference alpha-zero",
    "distribution. Main-text inference used the two prespecified item-level paired",
    "contrasts: User Likes minus No Preference and User Dislikes minus No Preference. The",
    "user-like minus user-dislike contrast was retained as a supplementary bipolar",
    "summary. Each contrast used a two-sided paired t test and a 5,000-resample",
    "item-bootstrap 95% confidence interval."
  ),
  "",
  paste(
    "Contrast estimates are shifts in no-preference standard-deviation units (Delta z),",
    "and paired standardized effects are reported as d_z. A supplementary omnibus",
    "test used Hotelling's T-squared on the two-dimensional paired-difference vector.",
    "This multivariate test allows an unstructured within-argument covariance matrix",
    "and does not require equal condition variances or sphericity. An unstructured",
    "repeated-measures GLS fit reproduced the paired estimates and standard errors as",
    "a numerical check."
  ),
  "",
  paste0(
    "Across all models, the largest absolute difference between an unstructured-GLS ",
    "standard error and its paired-analysis counterpart was ",
    format(max(feedback_state_diagnostics$max_abs_gls_minus_paired_se), scientific = TRUE),
    ", confirming the planned paired analysis."
  ),
  "",
  "### 2.2 Terminal Strong/Weak verdict",
  "",
  paste(
    "The parsed verdict was binary. The three related conditions were compared",
    "with Cochran's Q, which is the paired-binary chi-square test; an ordinary",
    "Pearson chi-square test of aggregated percentages would violate independence.",
    "The Q-based effect size was W = Q/[N(k-1)]. Planned pairwise comparisons used",
    "the exact McNemar/binomial test on discordant pairs. The primary pairwise",
    "effect size was the paired risk difference in percentage points with a",
    "5,000-resample item-bootstrap 95% confidence interval. A Haldane-corrected",
    "matched odds ratio (OR_m) is also tabulated."
  ),
  "",
  "### 2.3 VAA intervention curves",
  "",
  paste(
    "Normalized alpha was divided by the population SD of the fixed 11-level grid",
    "to obtain z_a. For each model, a binomial logistic mixed model estimated",
    "`verdict_strong ~ z_a * condition + (1 + z_a | item_id)`. A prespecified",
    "uncorrelated random-intercept/slope structure was used only when the correlated",
    "fit was singular or failed to converge. Simple slopes are reported as",
    "log-odds changes and odds ratios per one-SD increase in alpha. The interaction",
    "was tested by likelihood-ratio chi-square comparison."
  ),
  "",
  paste(
    "No Strong-versus-Weak candidate-logit contrast was recorded at a shared",
    "context. Generated-token log probabilities score only the continuation that",
    "was actually produced and were not analyzed as a continuous verdict endpoint.",
    "No mediation analysis was conducted."
  ),
  "",
  "## 3. Results",
  "",
  "### 3.1 Data completeness",
  "",
  paste(
    "All 296 arguments had complete VAA-state measurements in all three alpha-zero",
    "conditions for every model. Strictly parsed verdict complete cases ranged",
    "from 287 to 296 paired arguments. The intervention analysis retained 26,368",
    "of 26,400 rows (99.88%)."
  ),
  "",
  md_table(
    c("Model", "State items", "Verdict items", "Intervention valid rows"),
    sample_rows
  ),
  "",
  "### 3.2 Stated user preferences shifted the natural assistant-start VAA state",
  "",
  paste0(
    "For the primary Qwen2.5-14B model, user liking shifted the assistant-start VAA ",
    "state by ",
    fmt_est_ci(q14_like_state$estimate, q14_like_state$ci_low, q14_like_state$ci_high),
    " no-preference SD (d_z = ", fmt_num(q14_like_state$standardized_paired_effect_dz, 2),
    ", paired *t*(", round(q14_like_state$df), ") = ",
    fmt_num(q14_like_state$t_ratio, 2), ", *p* ", fmt_p(q14_like_state$p_value),
    "), whereas user disliking shifted it by ",
    fmt_est_ci(q14_dislike_state$estimate, q14_dislike_state$ci_low, q14_dislike_state$ci_high),
    " no-preference SD (d_z = ", fmt_num(q14_dislike_state$standardized_paired_effect_dz, 2),
    ", paired *t*(", round(q14_dislike_state$df), ") = ",
    fmt_num(q14_dislike_state$t_ratio, 2), ", *p* ",
    fmt_p(q14_dislike_state$p_value), ")."
  ),
  "",
  paste(
    "The User Dislikes minus No Preference contrast was negative and significant in all",
    "eight models. The User Likes minus No Preference contrast was positive and significant",
    "in seven models.",
    "Qwen2.5-72B was the sole state-level exception, with a near-zero estimate and",
    "a confidence interval spanning zero."
  ),
  "",
  "**Table 1. Planned paired VAA-state contrasts, Delta z [item-bootstrap 95% CI]**",
  "",
  md_table(
    c("Model", "User Likes - No Preference", "User Dislikes - No Preference", "User Likes - User Dislikes"),
    state_contrast_rows
  ),
  "",
  "**Supplementary Table 1. Omnibus repeated-measures robustness test**",
  "",
  md_table(c("Model", "Test", "partial eta^2", "p"), state_omnibus_rows),
  "",
  "### 3.3 Stated user preferences shifted the terminal verdict",
  "",
  paste0(
    "For Qwen2.5-14B, the three preference conditions differed in Strong-verdict ",
    "probability, *Q*(2) = ", fmt_num(q14_verdict$chi_square, 2),
    ", *p* ", fmt_p(q14_verdict$p_value),
    ", W = ", fmt_num(q14_verdict$q_effect_w, 3), ". Relative to No Preference, user ",
    "liking increased Strong verdicts by ",
    fmt_pct_ci(q14_like_verdict$probability_difference,
               q14_like_verdict$probability_ci_low,
               q14_like_verdict$probability_ci_high),
    " percentage points, exact *p* ", fmt_p(q14_like_verdict$p_value),
    "; user disliking decreased them by ",
    fmt_pct_ci(q14_dislike_verdict$probability_difference,
               q14_dislike_verdict$probability_ci_low,
               q14_dislike_verdict$probability_ci_high),
    " percentage points, exact *p* ", fmt_p(q14_dislike_verdict$p_value), "."
  ),
  "",
  paste(
    "Cochran's Q was significant in all eight models, and all 24 planned paired",
    "verdict contrasts were significant in their model-specific tests. User liking",
    "increased Strong verdicts in every model; user disliking decreased them in",
    "every model."
  ),
  "",
  "**Table 2. Omnibus preference effect on the binary verdict**",
  "",
  md_table(c("Model", "N", "Test", "W", "p"), verdict_omnibus_rows),
  "",
  "**Table 3. Planned verdict contrasts, risk difference in percentage points [95% CI]**",
  "",
  md_table(
    c("Model", "User Likes - No Preference", "User Dislikes - No Preference", "User Likes - User Dislikes"),
    verdict_contrast_rows
  ),
  "",
  "### 3.4 Direct VAA intervention retained a positive behavioral orientation",
  "",
  paste0(
    "For Qwen2.5-14B, all three condition-specific alpha slopes were positive and ",
    "significant (all *p* < .001). The alpha-by-preference-condition ",
    "interaction was also significant, likelihood-ratio chi-square(2) = ",
    fmt_num(q14_interaction$chi_square, 2), ", *p* ",
    fmt_p(q14_interaction$p_value), ". Thus, stated preference changed curve ",
    "location or steepness, but did not reverse the behavioral orientation of VAA."
  ),
  "",
  paste(
    "Across all models, all 24 model-by-condition simple slopes were positive and",
    "significant in their model-specific tests. Seven of eight alpha-by-preference-condition",
    "interaction tests were significant. The primary inference is the positive",
    "simple slope in every preference condition, not equality of the three curve shapes.",
    "Odds ratios are large because several curves approach deterministic Weak/Strong",
    "responses near the grid extremes; the log-odds slope and its confidence interval",
    "should therefore be reported alongside each OR."
  ),
  "",
  "**Table 4. Alpha-by-preference-condition interaction and model diagnostics**",
  "",
  md_table(
    c("Model", "Valid rows", "Random effects", "Interaction", "p"),
    intervention_interaction_rows
  ),
  "",
  "**Table 5. Condition-specific intervention slopes per one-SD alpha increase**",
  "",
  md_table(c("Model", "Condition", "b [95% CI]", "OR [95% CI]", "p"), intervention_slope_rows),
  "",
  "**Table 6. Differences between condition-specific alpha slopes**",
  "",
  md_table(
    c("Model", "User Likes - No Preference", "User Dislikes - No Preference", "User Likes - User Dislikes"),
    intervention_contrast_rows
  ),
  "",
  "## 4. Statistical conclusion and claim boundary",
  "",
  paste(
    "Stated user preferences shifted both the natural assistant-start VAA state and",
    "the later Strong/Weak evaluation. Direct intervention along the VAA increased",
    "the probability of a Strong verdict under every preference condition in every",
    "model. Together, these findings show that preference-induced sycophancy is",
    "reflected in the assistant-start VAA state while the behavioral orientation of",
    "VAA intervention is preserved."
  ),
  "",
  paste(
    "These analyses do not test statistical mediation or assume that a stated user",
    "preference is equivalent to a fixed intervention magnitude. Generated reasons",
    "were retained for audit but were not an analysis endpoint."
  ),
  "",
  "## 5. Reproducibility",
  "",
  paste(
    "Canonical script:",
    "`analysis/r/feedback_induced_sycophancy.R`."
  ),
  "",
  paste(
    "Machine-readable estimates, unadjusted model-specific p values, diagnostics,",
    "analysis metadata, and R session information are stored beside this report."
  )
)
writeLines(report, file.path(output_dir, "statistical_report.md"))

message("Wrote Preference-Induced Sycophancy results to: ", output_dir)

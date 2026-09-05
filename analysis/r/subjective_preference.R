#!/usr/bin/env Rscript

# Subjective Preference analysis using the reported content-free prompt.
#
# Visual output is intentionally excluded from this script. It writes compact,
# versionable statistical tables consumed by the Python Figure 3 workflow.

suppressPackageStartupMessages({
  library(lme4)
  library(lmerTest)
  library(emmeans)
  library(jsonlite)
})

options(contrasts = c("contr.sum", "contr.poly"))

args <- commandArgs(trailingOnly = TRUE)
script_arg <- grep("^--file=", commandArgs(), value = TRUE)
script_path <- normalizePath(sub("^--file=", "", script_arg[1]))
script_dir <- dirname(script_path)
repository_root <- normalizePath(file.path(script_dir, "..", ".."))

input_root <- if (length(args) >= 1) normalizePath(args[1]) else file.path(
  repository_root, "data", "processed", "subjective_preference"
)
output_dir <- if (length(args) >= 2) args[2] else file.path(
  repository_root, "results", "summaries", "subjective_preference"
)
n_perm <- if (length(args) >= 3) as.integer(args[3]) else 5000L
n_boot <- if (length(args) >= 4) as.integer(args[4]) else 5000L
seed <- if (length(args) >= 5) as.integer(args[5]) else 20260810L

dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

models <- c(
  "qwen25_3b", "qwen25_7b", "qwen25_14b", "qwen25_32b",
  "qwen25_72b", "llama3_8b", "mistral_7b", "gemma2_9b"
)
pair_levels <- c(
  "neutral_nonopposite", "neutral_opposite",
  "valenced_nonopposite", "valenced_opposite"
)
score_specs <- list(
  sequence = c(semantic = "semantic_component", position = "position_component"),
  first_token = c(
    semantic = "first_token_semantic_component",
    position = "first_token_position_component"
  )
)

bind_rows_base <- function(frames) {
  frames <- frames[!vapply(frames, is.null, logical(1))]
  if (length(frames) == 0) return(data.frame())
  columns <- unique(unlist(lapply(frames, names)))
  frames <- lapply(frames, function(frame) {
    missing <- setdiff(columns, names(frame))
    for (column in missing) frame[[column]] <- NA
    frame[columns]
  })
  do.call(rbind, frames)
}

ascii_lexicographically_first <- function(left, right) {
  left_code <- utf8ToInt(tolower(trimws(left)))
  right_code <- utf8ToInt(tolower(trimws(right)))
  shared_length <- min(length(left_code), length(right_code))
  if (shared_length > 0) {
    difference <- which(
      left_code[seq_len(shared_length)] != right_code[seq_len(shared_length)]
    )
    if (length(difference) > 0) {
      index <- difference[[1]]
      return(left_code[[index]] < right_code[[index]])
    }
  }
  length(left_code) <= length(right_code)
}

read_model_data <- function(model_name) {
  matches <- file.path(input_root, "trajectories", paste0(model_name, ".csv"))
  if (length(matches) != 1 || !file.exists(matches)) {
    stop(sprintf("%s: expected one processed file, found %d", model_name, length(matches)))
  }
  raw_matches <- file.path(input_root, "lexical_pairs", paste0(model_name, ".csv"))
  if (length(raw_matches) != 1 || !file.exists(raw_matches)) {
    stop(sprintf("%s: expected one raw file, found %d", model_name, length(raw_matches)))
  }
  frame <- read.csv(matches[1], stringsAsFactors = FALSE, check.names = FALSE)
  raw <- read.csv(raw_matches[1], stringsAsFactors = FALSE, check.names = FALSE)
  pair_map <- unique(raw[c("pair_id", "word_A", "word_B")])
  if (anyDuplicated(pair_map$pair_id)) {
    stop(sprintf("%s has inconsistent lexical metadata by pair_id", model_name))
  }
  required <- c(
    "model_name", "target_layer", "pair_id", "pair_class",
    "valence_status", "opposition_status", "alpha_norm", "template_name",
    unlist(score_specs)
  )
  missing <- setdiff(required, names(frame))
  if (length(missing) > 0) stop(paste(model_name, "missing:", paste(missing, collapse = ", ")))
  if (!all(frame$template_name == "preference_control")) {
    stop(sprintf("%s contains a non-preference_control row", model_name))
  }
  if (!all(frame$model_name == model_name)) stop(sprintf("%s model_name mismatch", model_name))
  if (anyDuplicated(frame[c("pair_id", "alpha_norm")])) {
    stop(sprintf("%s has duplicate pair-alpha rows", model_name))
  }
  alpha_grid <- sort(unique(frame$alpha_norm))
  expected_alpha <- seq(-1, 1, by = 0.2)
  if (length(alpha_grid) != 11 || max(abs(alpha_grid - expected_alpha)) > 1e-8) {
    stop(sprintf("%s has an unexpected alpha grid", model_name))
  }
  counts <- table(frame$pair_id)
  if (!all(counts == 11)) stop(sprintf("%s has incomplete pair trajectories", model_name))

  pair_index <- match(as.character(frame$pair_id), pair_map$pair_id)
  if (anyNA(pair_index)) stop(sprintf("%s has missing lexical metadata", model_name))
  frame$word_A_original <- pair_map$word_A[pair_index]
  frame$word_B_original <- pair_map$word_B[pair_index]
  frame$semantic_orientation_sign <- 1
  neutral <- frame$valence_status == "neutral"
  frame$semantic_orientation_sign[neutral] <- mapply(
    function(word_a, word_b) {
      if (ascii_lexicographically_first(word_a, word_b)) 1 else -1
    },
    frame$word_A_original[neutral],
    frame$word_B_original[neutral]
  )
  for (column in c("semantic_component", "first_token_semantic_component")) {
    frame[[column]] <- frame[[column]] * frame$semantic_orientation_sign
  }

  alpha_center <- mean(expected_alpha)
  alpha_scale <- sqrt(mean((expected_alpha - alpha_center)^2))
  frame$z_a <- (frame$alpha_norm - alpha_center) / alpha_scale
  frame$pair_id <- factor(frame$pair_id)
  frame$pair_class <- factor(frame$pair_class, levels = pair_levels)
  frame$valence_status <- factor(frame$valence_status, levels = c("neutral", "valenced"))
  frame$opposition_status <- factor(
    frame$opposition_status,
    levels = c("nonopposite", "opposite")
  )
  contrasts(frame$valence_status) <- contr.sum(2)
  contrasts(frame$opposition_status) <- contr.sum(2)
  frame
}

read_internal_data <- function(model_name) {
  matches <- file.path(
    input_root, "assistant_start_projection", paste0(model_name, ".csv")
  )
  if (length(matches) != 1 || !file.exists(matches)) {
    stop(sprintf("%s: expected one internal projection file, found %d", model_name, length(matches)))
  }
  frame <- read.csv(matches[1], stringsAsFactors = FALSE, check.names = FALSE)
  required <- c(
    "model_name", "target_layer", "pair_id", "pair_class",
    "valence_status", "opposition_status", "template_name",
    "word_A", "word_B",
    "vaa_projection_delta", "vaa_projection_unit_delta",
    "vaa_projection_cosine_delta"
  )
  missing <- setdiff(required, names(frame))
  if (length(missing) > 0) stop(paste(model_name, "internal missing:", paste(missing, collapse = ", ")))
  if (!all(frame$template_name == "preference_control")) {
    stop(sprintf("%s internal rows contain a non-preference_control prompt", model_name))
  }
  if (anyDuplicated(frame$pair_id)) stop(sprintf("%s has duplicate internal pair rows", model_name))
  frame$semantic_orientation_sign <- 1
  neutral <- frame$valence_status == "neutral"
  frame$semantic_orientation_sign[neutral] <- mapply(
    function(left, right) if (ascii_lexicographically_first(left, right)) 1 else -1,
    frame$word_A[neutral],
    frame$word_B[neutral]
  )
  for (column in c(
    "vaa_projection_delta", "vaa_projection_unit_delta",
    "vaa_projection_cosine_delta"
  )) {
    frame[[column]] <- frame[[column]] * frame$semantic_orientation_sign
  }
  frame
}

fit_lmer <- function(formula_text, data) {
  control <- lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 200000))
  correlated <- lmerTest::lmer(
    as.formula(formula_text), data = data, REML = TRUE, control = control
  )
  singular_correlated <- lme4::isSingular(correlated, tol = 1e-4)
  if (!singular_correlated) {
    return(list(model = correlated, formula = formula_text, singular = FALSE, fallback = FALSE))
  }
  fallback_formula <- gsub("\\(1 \\+ z_a \\| pair_id\\)", "(1 + z_a || pair_id)", formula_text)
  if (identical(fallback_formula, formula_text)) {
    return(list(model = correlated, formula = formula_text, singular = TRUE, fallback = FALSE))
  }
  uncorrelated <- lmerTest::lmer(
    as.formula(fallback_formula), data = data, REML = TRUE, control = control
  )
  list(
    model = uncorrelated,
    formula = fallback_formula,
    singular = lme4::isSingular(uncorrelated, tol = 1e-4),
    fallback = TRUE
  )
}

anova_table <- function(fit, model_name, score_mode, component, analysis) {
  table <- as.data.frame(anova(fit$model, type = 3, ddf = "Satterthwaite"))
  table$term <- rownames(table)
  rownames(table) <- NULL
  p_column <- grep("^Pr", names(table), value = TRUE)[1]
  out <- data.frame(
    model_name = model_name,
    score_mode = score_mode,
    component = component,
    analysis = analysis,
    term = table$term,
    numerator_df = table[["NumDF"]],
    denominator_df = table[["DenDF"]],
    f_value = table[["F value"]],
    p_value = table[[p_column]],
    formula_used = fit$formula,
    singular = fit$singular,
    used_uncorrelated_fallback = fit$fallback,
    stringsAsFactors = FALSE
  )
  out
}

trend_table <- function(fit, model_name, score_mode, component) {
  trends <- emtrends(
    fit$model,
    ~ valence_status * opposition_status,
    var = "z_a",
    lmer.df = "satterthwaite"
  )
  frame <- as.data.frame(confint(trends, level = 0.95))
  tests <- as.data.frame(test(trends))
  frame$p_value <- tests$p.value
  names(frame)[names(frame) == "z_a.trend"] <- "estimate"
  names(frame)[names(frame) == "lower.CL"] <- "ci_low"
  names(frame)[names(frame) == "upper.CL"] <- "ci_high"
  frame$model_name <- model_name
  frame$score_mode <- score_mode
  frame$component <- component
  frame$pair_class <- paste(frame$valence_status, frame$opposition_status, sep = "_")
  frame[c(
    "model_name", "score_mode", "component", "pair_class",
    "valence_status", "opposition_status", "estimate", "SE", "df",
    "ci_low", "ci_high", "p_value"
  )]
}

valence_marginal_trend_table <- function(fit, model_name, score_mode) {
  trends <- emtrends(
    fit$model,
    ~ valence_status,
    var = "z_a",
    infer = c(TRUE, TRUE),
    lmer.df = "satterthwaite"
  )
  frame <- as.data.frame(trends)
  names(frame)[names(frame) == "z_a.trend"] <- "estimate"
  names(frame)[names(frame) == "lower.CL"] <- "ci_low"
  names(frame)[names(frame) == "upper.CL"] <- "ci_high"
  names(frame)[names(frame) == "t.ratio"] <- "t_value"
  names(frame)[names(frame) == "p.value"] <- "p_value"
  frame$model_name <- model_name
  frame$score_mode <- score_mode
  frame$semantic_direction <- ifelse(
    frame$valence_status == "neutral",
    "alphabetically earlier minus later",
    "positive minus negative"
  )
  frame[c(
    "model_name", "score_mode", "valence_status", "semantic_direction",
    "estimate", "SE", "df", "ci_low", "ci_high", "t_value", "p_value"
  )]
}

planned_contrasts <- function(fit, model_name, score_mode, component) {
  trends <- emtrends(
    fit$model,
    ~ valence_status * opposition_status,
    var = "z_a",
    lmer.df = "satterthwaite"
  )
  valence_object <- contrast(
    trends, "revpairwise", by = "opposition_status", adjust = "holm"
  )
  opposition_object <- contrast(
    trends, "revpairwise", by = "valence_status", adjust = "holm"
  )
  valence <- as.data.frame(confint(valence_object))
  valence$p_value <- as.data.frame(test(valence_object))$p.value
  opposition <- as.data.frame(confint(opposition_object))
  opposition$p_value <- as.data.frame(test(opposition_object))$p.value
  normalize <- function(frame, family) {
    names(frame)[names(frame) == "lower.CL"] <- "ci_low"
    names(frame)[names(frame) == "upper.CL"] <- "ci_high"
    frame$model_name <- model_name
    frame$score_mode <- score_mode
    frame$component <- component
    frame$contrast_family <- family
    frame$contrast_stratum <- if (family == "valenced_minus_neutral_within_opposition") {
      as.character(frame$opposition_status)
    } else {
      as.character(frame$valence_status)
    }
    frame
  }
  bind_rows_base(list(
    normalize(valence, "valenced_minus_neutral_within_opposition"),
    normalize(opposition, "opposite_minus_nonopposite_within_valence")
  ))
}

fit_item_slopes <- function(frame, outcome, model_name, score_mode, component) {
  pieces <- split(frame, frame$pair_id)
  rows <- lapply(pieces, function(item) {
    fit <- lm(item[[outcome]] ~ item$z_a)
    coefficient <- summary(fit)$coefficients[2, ]
    data.frame(
      model_name = model_name,
      target_layer = item$target_layer[1],
      score_mode = score_mode,
      component = component,
      pair_id = as.character(item$pair_id[1]),
      pair_class = as.character(item$pair_class[1]),
      valence_status = as.character(item$valence_status[1]),
      opposition_status = as.character(item$opposition_status[1]),
      slope = unname(coefficient["Estimate"]),
      std_error = unname(coefficient["Std. Error"]),
      t_value = unname(coefficient["t value"]),
      p_value = unname(coefficient["Pr(>|t|)"]),
      stringsAsFactors = FALSE
    )
  })
  result <- do.call(rbind, rows)
  result$p_fdr_within_model_component <- p.adjust(result$p_value, method = "BH")
  result$abs_slope <- abs(result$slope)
  result
}

bootstrap_mean <- function(values, n_boot) {
  values <- as.numeric(values)
  estimates <- replicate(n_boot, mean(sample(values, length(values), replace = TRUE)))
  c(
    mean = mean(values),
    ci_low = unname(quantile(estimates, 0.025)),
    ci_high = unname(quantile(estimates, 0.975)),
    n_pairs = length(values)
  )
}

summarize_item_magnitude <- function(slopes, n_boot) {
  groups <- split(
    slopes,
    interaction(
      slopes$model_name, slopes$score_mode, slopes$component, slopes$pair_class,
      drop = TRUE
    )
  )
  rows <- lapply(groups, function(group) {
    estimate <- bootstrap_mean(group$abs_slope, n_boot)
    data.frame(
      model_name = group$model_name[1],
      score_mode = group$score_mode[1],
      component = group$component[1],
      pair_class = group$pair_class[1],
      valence_status = group$valence_status[1],
      opposition_status = group$opposition_status[1],
      mean_abs_slope = estimate["mean"],
      ci_low = estimate["ci_low"],
      ci_high = estimate["ci_high"],
      n_pairs = estimate["n_pairs"],
      stringsAsFactors = FALSE
    )
  })
  do.call(rbind, rows)
}

summarize_internal_projection <- function(data_by_model, n_boot) {
  metrics <- c(
    "vaa_projection_delta",
    "vaa_projection_unit_delta",
    "vaa_projection_cosine_delta"
  )
  rows <- list()
  index <- 1L
  for (model_name in names(data_by_model)) {
    frame <- data_by_model[[model_name]]
    for (metric in metrics) {
      groups <- split(frame, frame$pair_class)
      for (group in groups) {
        estimate <- bootstrap_mean(group[[metric]], n_boot)
        rows[[index]] <- data.frame(
          model_name = model_name,
          target_layer = group$target_layer[1],
          metric = metric,
          pair_class = group$pair_class[1],
          valence_status = group$valence_status[1],
          opposition_status = group$opposition_status[1],
          mean = estimate["mean"],
          ci_low = estimate["ci_low"],
          ci_high = estimate["ci_high"],
          n_pairs = estimate["n_pairs"],
          stringsAsFactors = FALSE
        )
        index <- index + 1L
      }
    }
  }
  do.call(rbind, rows)
}

fit_magnitude_model <- function(slopes, model_name, score_mode) {
  data <- slopes[slopes$model_name == model_name & slopes$score_mode == score_mode, ]
  data$pair_id <- factor(data$pair_id)
  data$component <- factor(data$component, levels = c("position", "semantic"))
  data$valence_status <- factor(data$valence_status, levels = c("neutral", "valenced"))
  data$opposition_status <- factor(
    data$opposition_status, levels = c("nonopposite", "opposite")
  )
  contrasts(data$component) <- contr.sum(2)
  contrasts(data$valence_status) <- contr.sum(2)
  contrasts(data$opposition_status) <- contr.sum(2)
  formula_text <- paste0(
    "log(abs_slope + 1e-6) ~ component * valence_status * opposition_status ",
    "+ (1 | pair_id)"
  )
  fit <- fit_lmer(formula_text, data)
  type3 <- anova_table(fit, model_name, score_mode, "semantic_vs_position", "magnitude")
  means <- emmeans(
    fit$model,
    ~ component | valence_status * opposition_status,
    lmer.df = "satterthwaite"
  )
  contrast_object <- contrast(means, "revpairwise", adjust = "holm")
  contrasts_out <- as.data.frame(confint(contrast_object))
  contrasts_out$p_value <- as.data.frame(test(contrast_object))$p.value
  names(contrasts_out)[names(contrasts_out) == "lower.CL"] <- "ci_low"
  names(contrasts_out)[names(contrasts_out) == "upper.CL"] <- "ci_high"
  contrasts_out$model_name <- model_name
  contrasts_out$score_mode <- score_mode
  contrasts_out$pair_class <- paste(
    contrasts_out$valence_status,
    contrasts_out$opposition_status,
    sep = "_"
  )
  list(type3 = type3, contrasts = contrasts_out)
}

fit_paired_magnitude_sensitivity <- function(slopes, model_name, score_mode) {
  data <- slopes[slopes$model_name == model_name & slopes$score_mode == score_mode, ]
  semantic <- data[data$component == "semantic", c(
    "pair_id", "pair_class", "valence_status", "opposition_status", "abs_slope"
  )]
  position <- data[data$component == "position", c("pair_id", "abs_slope")]
  names(semantic)[names(semantic) == "abs_slope"] <- "semantic_abs_slope"
  names(position)[names(position) == "abs_slope"] <- "position_abs_slope"
  if (anyDuplicated(semantic$pair_id) || anyDuplicated(position$pair_id)) {
    stop(sprintf("%s/%s: magnitude rows are not one-to-one by pair", model_name, score_mode))
  }
  paired <- merge(semantic, position, by = "pair_id")
  paired$log_magnitude_ratio <- log(paired$semantic_abs_slope + 1e-6) -
    log(paired$position_abs_slope + 1e-6)
  paired$valence_status <- factor(paired$valence_status, levels = c("neutral", "valenced"))
  paired$opposition_status <- factor(
    paired$opposition_status, levels = c("nonopposite", "opposite")
  )
  contrasts(paired$valence_status) <- contr.sum(2)
  contrasts(paired$opposition_status) <- contr.sum(2)
  fit <- lm(
    log_magnitude_ratio ~ valence_status * opposition_status,
    data = paired
  )
  means <- emmeans(fit, ~ valence_status * opposition_status)
  result <- as.data.frame(confint(means))
  result$p_value <- as.data.frame(test(means))$p.value
  names(result)[names(result) == "emmean"] <- "estimate"
  names(result)[names(result) == "lower.CL"] <- "ci_low"
  names(result)[names(result) == "upper.CL"] <- "ci_high"
  result$model_name <- model_name
  result$score_mode <- score_mode
  result$pair_class <- paste(
    result$valence_status, result$opposition_status, sep = "_"
  )
  result[c(
    "model_name", "score_mode", "pair_class", "valence_status",
    "opposition_status", "estimate", "SE", "df", "ci_low", "ci_high",
    "p_value"
  )]
}

permutation_null <- function(frame, model_name, n_perm) {
  frame <- frame[order(frame$pair_id, frame$z_a), ]
  pair_ids <- unique(as.character(frame$pair_id))
  n_alpha <- length(unique(frame$z_a))
  z <- sort(unique(frame$z_a))
  denominator <- sum(z^2)
  semantic <- matrix(frame$semantic_component, nrow = length(pair_ids), byrow = TRUE)
  position <- matrix(frame$position_component, nrow = length(pair_ids), byrow = TRUE)
  metadata <- frame[match(pair_ids, as.character(frame$pair_id)), c(
    "pair_id", "pair_class", "valence_status", "opposition_status"
  )]
  if (ncol(semantic) != n_alpha || n_alpha != 11) stop("Permutation matrix shape mismatch")

  group_key <- interaction(metadata$pair_class, drop = TRUE)
  group_levels <- levels(group_key)
  null_sem <- matrix(NA_real_, nrow = n_perm, ncol = length(group_levels))
  null_pos <- matrix(NA_real_, nrow = n_perm, ncol = length(group_levels))

  for (b in seq_len(n_perm)) {
    random_keys <- matrix(runif(length(pair_ids) * n_alpha), nrow = length(pair_ids))
    order_index <- t(apply(random_keys, 1, order))
    permuted_z <- matrix(z[order_index], nrow = length(pair_ids))
    sem_slopes <- rowSums(permuted_z * semantic) / denominator
    pos_slopes <- rowSums(permuted_z * position) / denominator
    null_sem[b, ] <- vapply(
      group_levels,
      function(level) mean(abs(sem_slopes[group_key == level])),
      numeric(1)
    )
    null_pos[b, ] <- vapply(
      group_levels,
      function(level) mean(abs(pos_slopes[group_key == level])),
      numeric(1)
    )
  }

  observed_sem <- vapply(
    group_levels,
    function(level) mean(abs(rowSums(matrix(z, nrow = length(pair_ids), ncol = n_alpha, byrow = TRUE) * semantic) / denominator)[group_key == level]),
    numeric(1)
  )
  observed_pos <- vapply(
    group_levels,
    function(level) mean(abs(rowSums(matrix(z, nrow = length(pair_ids), ncol = n_alpha, byrow = TRUE) * position) / denominator)[group_key == level]),
    numeric(1)
  )

  make_rows <- function(null_matrix, observed, component) {
    do.call(rbind, lapply(seq_along(group_levels), function(index) {
      values <- null_matrix[, index]
      pair_class <- as.character(metadata$pair_class[which(group_key == group_levels[index])[1]])
      data.frame(
        model_name = model_name,
        score_mode = "sequence",
        component = component,
        pair_class = pair_class,
        valence_status = as.character(metadata$valence_status[which(group_key == group_levels[index])[1]]),
        opposition_status = as.character(metadata$opposition_status[which(group_key == group_levels[index])[1]]),
        observed_mean_abs_slope = observed[index],
        null_mean = mean(values),
        null_ci_low = unname(quantile(values, 0.025)),
        null_ci_high = unname(quantile(values, 0.975)),
        p_value = (1 + sum(values >= observed[index])) / (n_perm + 1),
        n_permutations = n_perm,
        stringsAsFactors = FALSE
      )
    }))
  }
  bind_rows_base(list(
    make_rows(null_sem, observed_sem, "semantic"),
    make_rows(null_pos, observed_pos, "position")
  ))
}

trajectory_summary <- function(frame, model_name, n_boot) {
  rows <- list()
  index <- 1L
  for (score_mode in names(score_specs)) {
    for (component in names(score_specs[[score_mode]])) {
      outcome <- score_specs[[score_mode]][[component]]
      groups <- split(
        frame,
        interaction(frame$pair_class, frame$alpha_norm, drop = TRUE)
      )
      for (group in groups) {
        estimate <- bootstrap_mean(group[[outcome]], n_boot)
        rows[[index]] <- data.frame(
          model_name = model_name,
          score_mode = score_mode,
          component = component,
          pair_class = as.character(group$pair_class[1]),
          valence_status = as.character(group$valence_status[1]),
          opposition_status = as.character(group$opposition_status[1]),
          alpha_norm = group$alpha_norm[1],
          mean = estimate["mean"],
          ci_low = estimate["ci_low"],
          ci_high = estimate["ci_high"],
          n_pairs = estimate["n_pairs"],
          stringsAsFactors = FALSE
        )
        index <- index + 1L
      }
    }
  }
  do.call(rbind, rows)
}

adjust_across_models <- function(frame, grouping_columns, p_column = "p_value") {
  frame$p_holm_across_models <- NA_real_
  groups <- interaction(frame[grouping_columns], drop = TRUE, lex.order = TRUE)
  for (group in unique(groups)) {
    indexes <- which(groups == group & !is.na(frame[[p_column]]))
    frame$p_holm_across_models[indexes] <- p.adjust(frame[[p_column]][indexes], method = "holm")
  }
  frame
}

set.seed(seed)
data_by_model <- setNames(lapply(models, read_model_data), models)
internal_by_model <- setNames(lapply(models, read_internal_data), models)

neutral_orientation_audit <- do.call(rbind, lapply(models, function(model_name) {
  frame <- data_by_model[[model_name]]
  neutral <- frame[frame$valence_status == "neutral", c(
    "pair_id", "pair_class", "word_A_original", "word_B_original",
    "semantic_orientation_sign"
  )]
  neutral <- unique(neutral)
  neutral$model_name <- model_name
  neutral$analysis_word_A <- ifelse(
    neutral$semantic_orientation_sign == 1,
    neutral$word_A_original,
    neutral$word_B_original
  )
  neutral$analysis_word_B <- ifelse(
    neutral$semantic_orientation_sign == 1,
    neutral$word_B_original,
    neutral$word_A_original
  )
  neutral[c(
    "model_name", "pair_id", "pair_class", "word_A_original", "word_B_original",
    "analysis_word_A", "analysis_word_B", "semantic_orientation_sign"
  )]
}))

input_counts <- do.call(rbind, lapply(models, function(model_name) {
  frame <- data_by_model[[model_name]]
  class_counts <- as.data.frame(table(frame$pair_class) / 11)
  names(class_counts) <- c("pair_class", "n_pairs")
  class_counts$model_name <- model_name
  class_counts$target_layer <- unique(frame$target_layer)
  class_counts$n_rows <- nrow(frame)
  class_counts$template_name <- unique(frame$template_name)
  class_counts
}))

type3_rows <- list()
trend_rows <- list()
contrast_rows <- list()
item_slope_rows <- list()
fit_status_rows <- list()
trajectory_rows <- list()
semantic_valence_trend_rows <- list()
index <- 1L

for (model_name in models) {
  message("Fitting alpha-level models: ", model_name)
  frame <- data_by_model[[model_name]]
  trajectory_rows[[model_name]] <- trajectory_summary(frame, model_name, n_boot)
  for (score_mode in names(score_specs)) {
    for (component in names(score_specs[[score_mode]])) {
      outcome <- score_specs[[score_mode]][[component]]
      formula_text <- paste0(
        outcome,
        " ~ z_a * valence_status * opposition_status + (1 + z_a | pair_id)"
      )
      fit <- fit_lmer(formula_text, frame)
      type3_rows[[index]] <- anova_table(
        fit, model_name, score_mode, component, "alpha_level"
      )
      trend_rows[[index]] <- trend_table(fit, model_name, score_mode, component)
      if (component == "semantic") {
        semantic_valence_trend_rows[[paste(model_name, score_mode, sep = "::")]] <-
          valence_marginal_trend_table(fit, model_name, score_mode)
      }
      contrast_rows[[index]] <- planned_contrasts(
        fit, model_name, score_mode, component
      )
      item_slope_rows[[index]] <- fit_item_slopes(
        frame, outcome, model_name, score_mode, component
      )
      fit_status_rows[[index]] <- data.frame(
        model_name = model_name,
        score_mode = score_mode,
        component = component,
        formula_used = fit$formula,
        singular = fit$singular,
        used_uncorrelated_fallback = fit$fallback,
        n_rows = nrow(frame),
        n_pairs = nlevels(frame$pair_id),
        stringsAsFactors = FALSE
      )
      index <- index + 1L
    }
  }
}

type3 <- bind_rows_base(type3_rows)
trends <- bind_rows_base(trend_rows)
contrasts_out <- bind_rows_base(contrast_rows)
item_slopes <- bind_rows_base(item_slope_rows)
fit_status <- bind_rows_base(fit_status_rows)
trajectories <- bind_rows_base(trajectory_rows)
semantic_valence_trends <- bind_rows_base(semantic_valence_trend_rows)

type3 <- adjust_across_models(type3, c("score_mode", "component", "term"))
trends <- adjust_across_models(trends, c("score_mode", "component", "pair_class"))
contrasts_out <- adjust_across_models(
  contrasts_out,
  c("score_mode", "component", "contrast_family", "contrast_stratum")
)
semantic_valence_trends <- adjust_across_models(
  semantic_valence_trends, c("score_mode", "valence_status")
)

magnitude_type3 <- list()
magnitude_contrasts <- list()
paired_magnitude_rows <- list()
index <- 1L
for (model_name in models) {
  for (score_mode in names(score_specs)) {
    message("Fitting magnitude model: ", model_name, " / ", score_mode)
    fit <- fit_magnitude_model(item_slopes, model_name, score_mode)
    magnitude_type3[[index]] <- fit$type3
    magnitude_contrasts[[index]] <- fit$contrasts
    paired_magnitude_rows[[index]] <- fit_paired_magnitude_sensitivity(
      item_slopes, model_name, score_mode
    )
    index <- index + 1L
  }
}
magnitude_type3 <- bind_rows_base(magnitude_type3)
magnitude_contrasts <- bind_rows_base(magnitude_contrasts)
paired_magnitude <- bind_rows_base(paired_magnitude_rows)
magnitude_type3 <- adjust_across_models(
  magnitude_type3, c("score_mode", "term")
)
magnitude_contrasts <- adjust_across_models(
  magnitude_contrasts, c("score_mode", "pair_class")
)
paired_magnitude <- adjust_across_models(
  paired_magnitude, c("score_mode", "pair_class")
)

message("Bootstrapping item-level magnitude summaries")
magnitude_summary <- summarize_item_magnitude(item_slopes, n_boot)
message("Bootstrapping corrected internal order-flip summaries")
internal_summary <- summarize_internal_projection(internal_by_model, n_boot)

permutation_rows <- list()
for (model_name in models) {
  message("Permutation null: ", model_name)
  permutation_rows[[model_name]] <- permutation_null(
    data_by_model[[model_name]], model_name, n_perm
  )
}
permutation_summary <- bind_rows_base(permutation_rows)
permutation_summary$p_fdr_within_model <- ave(
  permutation_summary$p_value,
  permutation_summary$model_name,
  FUN = function(values) p.adjust(values, method = "BH")
)
permutation_summary <- adjust_across_models(
  permutation_summary, c("component", "pair_class")
)

write.csv(input_counts, file.path(output_dir, "input_counts.csv"), row.names = FALSE)
write.csv(
  neutral_orientation_audit,
  file.path(output_dir, "neutral_semantic_orientation_audit.csv"),
  row.names = FALSE
)
write.csv(fit_status, file.path(output_dir, "model_fit_status.csv"), row.names = FALSE)
write.csv(type3, file.path(output_dir, "alpha_level_type3_tests.csv"), row.names = FALSE)
write.csv(trends, file.path(output_dir, "model_condition_slopes.csv"), row.names = FALSE)
write.csv(
  semantic_valence_trends,
  file.path(output_dir, "semantic_valence_marginal_trends.csv"),
  row.names = FALSE
)
write.csv(contrasts_out, file.path(output_dir, "planned_slope_contrasts.csv"), row.names = FALSE)
write.csv(item_slopes, file.path(output_dir, "item_level_slopes.csv"), row.names = FALSE)
write.csv(magnitude_type3, file.path(output_dir, "magnitude_type3_tests.csv"), row.names = FALSE)
write.csv(
  magnitude_contrasts,
  file.path(output_dir, "magnitude_semantic_position_contrasts.csv"),
  row.names = FALSE
)
write.csv(
  paired_magnitude,
  file.path(output_dir, "magnitude_paired_log_ratio_sensitivity.csv"),
  row.names = FALSE
)
write.csv(
  magnitude_summary,
  file.path(output_dir, "magnitude_condition_summary.csv"),
  row.names = FALSE
)
write.csv(
  permutation_summary,
  file.path(output_dir, "permutation_null_summary.csv"),
  row.names = FALSE
)
write.csv(
  trajectories,
  file.path(output_dir, "trajectory_summary.csv"),
  row.names = FALSE
)
write.csv(
  internal_summary,
  file.path(output_dir, "internal_projection_condition_summary.csv"),
  row.names = FALSE
)

metadata <- list(
  analysis = "Subjective Preference regression",
  input_root = sub(paste0("^", repository_root, "/"), "", normalizePath(input_root)),
  output_dir = sub(paste0("^", repository_root, "/"), "", normalizePath(output_dir)),
  models = models,
  primary_score = "full-sequence candidate log-probability margin",
  robustness_score = "first-token candidate logit difference",
  alpha_grid = seq(-1, 1, by = 0.2),
  z_alpha_definition = paste0(
    "alpha_norm centered and divided by the population SD of the fixed ",
    "11-level grid (sqrt(mean((alpha - mean(alpha))^2)))"
  ),
  semantic_orientation = list(
    valenced = "positive minus negative",
    neutral = "ASCII case-insensitive alphabetically earlier minus later",
    scope = paste(
      "semantic components and final-input-token AB/BA projection deltas;",
      "position components are unchanged"
    )
  ),
  n_permutations = n_perm,
  n_bootstrap = n_boot,
  seed = seed,
  prompt_template = "XXXXXXXXXXXXXXXXXXXXXX, {option1} or {option2}? Please answer only with {option1} or {option2}.",
  inference_unit = "lexical pair within each model; models analyzed separately",
  p_value_reporting = paste(
    "Unadjusted model-specific p values are primary; Holm-adjusted columns are",
    "retained as a cross-model sensitivity summary"
  )
)
write_json(
  metadata,
  file.path(output_dir, "analysis_metadata.json"),
  pretty = TRUE,
  auto_unbox = TRUE
)
session_info <- sub("[[:space:]]+$", "", capture.output(sessionInfo()))
conda_prefix <- Sys.getenv("CONDA_PREFIX", unset = "")
if (nzchar(conda_prefix)) {
  session_info <- gsub(
    normalizePath(conda_prefix), "$CONDA_PREFIX", session_info, fixed = TRUE
  )
}
writeLines(session_info, file.path(output_dir, "r_session_info.txt"))

message("Subjective Preference analysis complete: ", normalizePath(output_dir))

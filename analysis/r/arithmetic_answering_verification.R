#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(jsonlite)
})

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
  file.path(repository_root, "data", "processed", "arithmetic_answering_verification", "item_slopes")
}
output_dir <- if (length(output_arg)) {
  normalizePath(sub("^--output-dir=", "", output_arg[[1]]), mustWork = FALSE)
} else {
  file.path(repository_root, "results", "summaries", "arithmetic_answering_verification")
}
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

models <- c(
  "qwen25_3b", "qwen25_7b", "llama3_8b", "mistral_7b",
  "gemma2_9b", "qwen25_14b", "qwen25_32b", "qwen25_72b"
)
condition_levels <- c(
  "direct_numeric", "verification_true", "verification_false"
)
condition_labels <- c(
  direct_numeric = "Direct Numeric",
  verification_true = "True Statement",
  verification_false = "False Statement"
)
zero_tolerance <- 1e-12

write_table <- function(data, filename) {
  write.csv(data, file.path(output_dir, filename), row.names = FALSE, na = "")
}

one_sample_slope <- function(values, model, condition) {
  n_items <- length(values)
  estimate <- mean(values)
  slope_sd <- sd(values)

  if (slope_sd <= zero_tolerance && abs(estimate) <= zero_tolerance) {
    return(data.frame(
      model_name = model,
      condition = condition,
      condition_label = unname(condition_labels[[condition]]),
      n_items = n_items,
      estimate = 0,
      item_slope_sd = 0,
      se = 0,
      df = n_items - 1,
      ci_low = 0,
      ci_high = 0,
      t_ratio = NA_real_,
      p_value = 1,
      standardized_effect_dz = 0,
      inference_note = "All item-level accuracy slopes were zero",
      stringsAsFactors = FALSE
    ))
  }

  test <- t.test(values, mu = 0)
  data.frame(
    model_name = model,
    condition = condition,
    condition_label = unname(condition_labels[[condition]]),
    n_items = n_items,
    estimate = estimate,
    item_slope_sd = slope_sd,
    se = slope_sd / sqrt(n_items),
    df = unname(test$parameter),
    ci_low = unname(test$conf.int[[1]]),
    ci_high = unname(test$conf.int[[2]]),
    t_ratio = unname(test$statistic),
    p_value = test$p.value,
    standardized_effect_dz = estimate / slope_sd,
    inference_note = "One-sample test of item-level accuracy slopes",
    stringsAsFactors = FALSE
  )
}

paired_slope_contrast <- function(wide, model, contrast, left, right) {
  difference <- wide[[left]] - wide[[right]]
  test <- t.test(difference, mu = 0)
  difference_sd <- sd(difference)
  data.frame(
    model_name = model,
    contrast = contrast,
    n_pairs = length(difference),
    estimate = mean(difference),
    paired_sd = difference_sd,
    se = difference_sd / sqrt(length(difference)),
    df = unname(test$parameter),
    ci_low = unname(test$conf.int[[1]]),
    ci_high = unname(test$conf.int[[2]]),
    t_ratio = unname(test$statistic),
    p_value = test$p.value,
    standardized_paired_effect_dz = mean(difference) / difference_sd,
    stringsAsFactors = FALSE
  )
}

estimate_rows <- list()
contrast_rows <- list()
input_rows <- list()
combined_rows <- list()

for (model in models) {
  files <- file.path(input_dir, paste0(model, ".csv"))
  if (length(files) != 1L || !file.exists(files)) {
    stop("Expected one item-slope file for ", model, "; found ", length(files))
  }

  data <- read.csv(files[[1]], stringsAsFactors = FALSE)
  required <- c("model_name", "item_id", "mode", "candidate_accuracy_slope")
  missing <- setdiff(required, names(data))
  if (length(missing)) stop("Missing columns for ", model, ": ", paste(missing, collapse = ", "))
  data <- data[data$mode %in% condition_levels, ]
  data$condition <- factor(data$mode, levels = condition_levels)

  counts <- table(data$item_id)
  if (length(counts) != 150L || any(counts != length(condition_levels))) {
    stop(model, " does not contain 150 complete three-condition items")
  }

  duplicate_count <- sum(duplicated(data[c("item_id", "condition")]))
  if (duplicate_count) stop(model, " contains duplicated item-condition rows")

  wide <- reshape(
    data[c("item_id", "condition", "candidate_accuracy_slope")],
    idvar = "item_id", timevar = "condition", direction = "wide"
  )
  names(wide) <- sub("^candidate_accuracy_slope\\.", "", names(wide))
  wide <- wide[order(wide$item_id), ]

  estimates <- do.call(rbind, lapply(condition_levels, function(condition) {
    one_sample_slope(wide[[condition]], model, condition)
  }))
  contrasts <- rbind(
    paired_slope_contrast(
      wide, model, "true_statement_minus_direct_numeric",
      "verification_true", "direct_numeric"
    ),
    paired_slope_contrast(
      wide, model, "false_statement_minus_direct_numeric",
      "verification_false", "direct_numeric"
    ),
    paired_slope_contrast(
      wide, model, "true_statement_minus_false_statement",
      "verification_true", "verification_false"
    )
  )

  estimate_rows[[model]] <- estimates
  contrast_rows[[model]] <- contrasts
  input_rows[[model]] <- data.frame(
    model_name = model,
    target_layer = unique(data$target_layer),
    input_file = sub(paste0("^", repository_root, "/"), "", normalizePath(files[[1]])),
    n_items = length(unique(data$item_id)),
    n_conditions = length(unique(data$condition)),
    n_item_condition_rows = nrow(data),
    stringsAsFactors = FALSE
  )
  combined_rows[[model]] <- data
}

condition_estimates <- do.call(rbind, estimate_rows)
condition_contrasts <- do.call(rbind, contrast_rows)
input_audit <- do.call(rbind, input_rows)
combined_data <- do.call(rbind, combined_rows)

write_table(condition_estimates, "accuracy_slope_condition_estimates.csv")
write_table(condition_contrasts, "accuracy_slope_paired_contrasts.csv")
write_table(input_audit, "input_audit.csv")

qwen14 <- condition_estimates[condition_estimates$model_name == "qwen25_14b", ]
qwen14_contrasts <- condition_contrasts[
  condition_contrasts$model_name == "qwen25_14b",
]
write_table(qwen14, "qwen25_14b_accuracy_slope_estimates.csv")
write_table(qwen14_contrasts, "qwen25_14b_accuracy_slope_contrasts.csv")

metadata <- list(
  analysis = "Arithmetic Answering and Verification candidate-accuracy slopes",
  analysis_unit = "arithmetic item",
  manuscript_endpoint = "candidate accuracy",
  internal_only_endpoint = "continuous candidate log-probability margin",
  alpha_grid = seq(-1, 1, by = 0.2),
  conditions = unname(condition_labels),
  n_models = length(models),
  n_items_per_model = 150,
  inference = list(
    condition = "two-sided one-sample t-test of item-level alpha slopes",
    framing_contrast = "two-sided paired t-test of item-level alpha slopes",
    zero_variance_rule = paste(
      "When every item-level slope is numerically zero, report estimate and CI",
      "as zero and p=1 instead of fitting a singular residual-variance model."
    ),
    multiplicity = "No cross-model multiplicity adjustment; each model is a replication unit."
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

cat("Wrote Arithmetic Answering and Verification analysis to", output_dir, "\n")
cat("Models:", length(models), " Items per model: 150\n")
print(qwen14[, c(
  "condition_label", "estimate", "ci_low", "ci_high", "t_ratio", "df", "p_value"
)], row.names = FALSE)

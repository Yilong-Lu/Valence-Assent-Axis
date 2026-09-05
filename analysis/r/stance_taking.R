#!/usr/bin/env Rscript

# Stance and Sound-Reasoning analyses reported in Figure 5b-c.

suppressPackageStartupMessages({
  library(lme4)
  library(lmerTest)
})

script_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
if (length(script_arg) != 1L) stop("Could not resolve script path")
script_path <- normalizePath(sub("^--file=", "", script_arg))
repository_root <- normalizePath(file.path(dirname(script_path), "..", ".."))

args <- commandArgs(trailingOnly = TRUE)
input_arg <- grep("^--input=", args, value = TRUE)
output_arg <- grep("^--output=", args, value = TRUE)
input_path <- if (length(input_arg)) {
  normalizePath(sub("^--input=", "", input_arg[[1]]))
} else {
  file.path(repository_root, "data", "source_data", "figure5_stance_qwen25_14b.csv")
}
output_path <- if (length(output_arg)) {
  normalizePath(sub("^--output=", "", output_arg[[1]]), mustWork = FALSE)
} else {
  file.path(repository_root, "results", "summaries", "stance_taking.csv")
}
dir.create(dirname(output_path), recursive = TRUE, showWarnings = FALSE)

data <- read.csv(input_path, stringsAsFactors = FALSE, check.names = FALSE)
required <- c(
  "statement", "z_a", "z_abs_a", "answer_stance", "reasoning_stance",
  "response_type"
)
missing <- setdiff(required, names(data))
if (length(missing)) stop("Missing columns: ", paste(missing, collapse = ", "))

data$answer_stance_z <- as.numeric(scale(data$answer_stance))
data$reasoning_stance_z <- as.numeric(scale(data$reasoning_stance))
data$is_sound_reasoning <- as.integer(data$response_type == "Sound Reasoning")
data$statement <- factor(data$statement)

linear_result <- function(outcome, endpoint) {
  frame <- data[complete.cases(data[c(outcome, "z_a", "statement")]), ]
  fit <- lmer(
    as.formula(paste0(outcome, " ~ z_a + (1 | statement)")),
    data = frame,
    REML = TRUE
  )
  estimate <- summary(fit)$coefficients["z_a", ]
  coefficient <- unname(estimate[["Estimate"]])
  standard_error <- unname(estimate[["Std. Error"]])
  statistic <- coefficient / standard_error
  data.frame(
    endpoint = endpoint,
    model_family = "linear mixed-effects",
    predictor = "z_a",
    coefficient = coefficient,
    standard_error = standard_error,
    statistic = statistic,
    p_value = 2 * pnorm(-abs(statistic)),
    ci_low = coefficient - qnorm(0.975) * standard_error,
    ci_high = coefficient + qnorm(0.975) * standard_error,
    n_rows = nrow(frame),
    n_items = length(unique(frame$statement)),
    stringsAsFactors = FALSE
  )
}

rows <- list(
  linear_result("answer_stance_z", "Answer Stance"),
  linear_result("reasoning_stance_z", "Reasoning Stance")
)

sound_fit <- glmer(
  is_sound_reasoning ~ z_abs_a + (1 | statement),
  data = data,
  family = binomial,
  control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 200000))
)
sound <- summary(sound_fit)$coefficients["z_abs_a", ]
sound_coefficient <- unname(sound[["Estimate"]])
sound_se <- unname(sound[["Std. Error"]])
rows[[3]] <- data.frame(
  endpoint = "Sound Reasoning",
  model_family = "binomial mixed-effects",
  predictor = "z_abs_a",
  coefficient = sound_coefficient,
  standard_error = sound_se,
  statistic = unname(sound[["z value"]]),
  p_value = unname(sound[["Pr(>|z|)"]]),
  ci_low = sound_coefficient - qnorm(0.975) * sound_se,
  ci_high = sound_coefficient + qnorm(0.975) * sound_se,
  n_rows = nrow(data),
  n_items = length(unique(data$statement)),
  stringsAsFactors = FALSE
)

write.csv(do.call(rbind, rows), output_path, row.names = FALSE, na = "")
message("Wrote Stance-Taking analysis to: ", output_path)

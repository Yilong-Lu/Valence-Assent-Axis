#!/usr/bin/env Rscript

# Cross-model analysis of coherent hallucinations in Factual Judgment.

suppressPackageStartupMessages(library(lme4))

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
  file.path(repository_root, "data", "processed", "original", "cross_model_factual_judgment.csv")
}
output_path <- if (length(output_arg)) {
  normalizePath(sub("^--output=", "", output_arg[[1]]), mustWork = FALSE)
} else {
  file.path(repository_root, "results", "summaries", "cross_model_factual_judgment.csv")
}
dir.create(dirname(output_path), recursive = TRUE, showWarnings = FALSE)

data <- read.csv(input_path, stringsAsFactors = FALSE, check.names = FALSE)
required <- c("model", "QID", "alignment_pressure", "Is_Coherent_Hallucination")
missing <- setdiff(required, names(data))
if (length(missing)) stop("Missing columns: ", paste(missing, collapse = ", "))

models <- sort(unique(data$model))
data$alignment_pressure_z <- as.numeric(scale(data$alignment_pressure))
rows <- lapply(models, function(model_name) {
  frame <- data[data$model == model_name, ]
  frame$QID <- factor(frame$QID)
  fit <- glmer(
    Is_Coherent_Hallucination ~ alignment_pressure_z + (1 | QID),
    data = frame,
    family = binomial,
    control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 200000))
  )
  estimate <- summary(fit)$coefficients["alignment_pressure_z", ]
  interval <- suppressMessages(confint(
    fit, parm = "alignment_pressure_z", method = "Wald", level = 0.95
  ))
  data.frame(
    model_name = model_name,
    predictor = "alignment_pressure_z",
    coefficient = unname(estimate[["Estimate"]]),
    statistic = unname(estimate[["z value"]]),
    p_value = unname(estimate[["Pr(>|z|)"]]),
    ci_low = unname(interval[[1]]),
    ci_high = unname(interval[[2]]),
    n_rows = nrow(frame),
    n_items = length(unique(frame$QID)),
    stringsAsFactors = FALSE
  )
})

output <- do.call(rbind, rows)
write.csv(output, output_path, row.names = FALSE, na = "")
message("Wrote cross-model Factual Judgment analysis to: ", output_path)

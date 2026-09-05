#!/usr/bin/env Rscript

# Answer accuracy and Bayesian reasoning-pattern analyses reported in Figure 4.

suppressPackageStartupMessages(library(lme4))

script_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
if (length(script_arg) != 1L) stop("Could not resolve script path")
script_path <- normalizePath(sub("^--file=", "", script_arg))
repository_root <- normalizePath(file.path(dirname(script_path), "..", ".."))

args <- commandArgs(trailingOnly = TRUE)
input_arg <- grep("^--input-dir=", args, value = TRUE)
output_arg <- grep("^--output-dir=", args, value = TRUE)
run_bayesian <- "--fit-bayesian" %in% args
input_dir <- if (length(input_arg)) {
  normalizePath(sub("^--input-dir=", "", input_arg[[1]]))
} else {
  file.path(repository_root, "data", "source_data")
}
output_dir <- if (length(output_arg)) {
  normalizePath(sub("^--output-dir=", "", output_arg[[1]]), mustWork = FALSE)
} else {
  file.path(repository_root, "results", "summaries", "reasoning_subordination")
}
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

task_files <- c(
  alphabetical_think_then_answer = "figure4_alphabetical_think_then_answer.csv",
  alphabetical_answer_then_think = "figure4_alphabetical_answer_then_think.csv",
  factual_judgment = "figure4_factual_judgment.csv"
)
task_labels <- c(
  alphabetical_think_then_answer = "Alphabetical Order: Think-then-Answer",
  alphabetical_answer_then_think = "Alphabetical Order: Answer-then-Think",
  factual_judgment = "Factual Judgment"
)

read_task <- function(task) {
  path <- file.path(input_dir, task_files[[task]])
  if (!file.exists(path)) stop("Missing input: ", path)
  frame <- read.csv(path, stringsAsFactors = FALSE, check.names = FALSE)
  required <- c("QID", "alignment_pressure", "correct", "response_type")
  missing <- setdiff(required, names(frame))
  if (length(missing)) stop(task, " missing columns: ", paste(missing, collapse = ", "))
  frame$alignment_pressure_z <- as.numeric(scale(frame$alignment_pressure))
  frame$QID <- factor(frame$QID)
  frame
}

accuracy_rows <- lapply(names(task_files), function(task) {
  frame <- read_task(task)
  fit <- glm(correct ~ alignment_pressure_z, data = frame, family = binomial)
  estimate <- summary(fit)$coefficients["alignment_pressure_z", ]
  coefficient <- unname(estimate[["Estimate"]])
  standard_error <- unname(estimate[["Std. Error"]])
  interval <- suppressMessages(confint(fit, parm = "alignment_pressure_z"))
  data.frame(
    task = task_labels[[task]],
    coefficient = coefficient,
    standard_error = standard_error,
    statistic = unname(estimate[["z value"]]),
    p_value = unname(estimate[["Pr(>|z|)"]]),
    ci_low = unname(interval[[1]]),
    ci_high = unname(interval[[2]]),
    odds_ratio = exp(coefficient),
    n_rows = nrow(frame),
    n_items = length(unique(frame$QID)),
    stringsAsFactors = FALSE
  )
})
write.csv(
  do.call(rbind, accuracy_rows),
  file.path(output_dir, "answer_accuracy.csv"),
  row.names = FALSE,
  na = ""
)

if (run_bayesian) {
  if (!requireNamespace("brms", quietly = TRUE)) {
    stop("The --fit-bayesian option requires the R package 'brms'")
  }
  reasoning_levels <- c(
    "Sound Reasoning", "Coherent Hallucination",
    "Incoherent Hallucination", "Contradictory Reasoning"
  )
  posterior_rows <- list()
  index <- 1L
  for (task in names(task_files)) {
    frame <- read_task(task)
    frame <- frame[frame$response_type %in% reasoning_levels, ]
    frame$response_type <- factor(frame$response_type, levels = reasoning_levels)
    fit <- brms::brm(
      response_type ~ alignment_pressure_z + (1 | QID),
      data = frame,
      family = brms::categorical(link = "logit"),
      chains = 4,
      cores = min(4L, parallel::detectCores()),
      iter = 2000,
      warmup = 1000,
      seed = 20260831,
      refresh = 100
    )
    saveRDS(fit, file.path(output_dir, paste0(task, "_brms_fit.rds")))
    fixed <- as.data.frame(brms::fixef(fit, probs = c(0.025, 0.975)))
    fixed$term <- rownames(fixed)
    fixed <- fixed[grepl("alignment_pressure_z$", fixed$term), ]
    fixed$task <- task_labels[[task]]
    posterior_rows[[index]] <- fixed[
      c("task", "term", "Estimate", "Est.Error", "Q2.5", "Q97.5")
    ]
    index <- index + 1L
  }
  write.csv(
    do.call(rbind, posterior_rows),
    file.path(output_dir, "reasoning_pattern_coefficients.csv"),
    row.names = FALSE,
    na = ""
  )
}

message("Wrote Figure 4 analyses to: ", output_dir)
if (!run_bayesian) {
  message("Bayesian reasoning-pattern models were not run; add --fit-bayesian to fit them.")
}

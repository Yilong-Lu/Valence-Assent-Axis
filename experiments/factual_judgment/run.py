"""Run the Factual Judgment task."""

from experiments.generative_reasoning import parse_args, run_task


if __name__ == "__main__":
    run_task("factual_judgment", parse_args("factual_judgment"))

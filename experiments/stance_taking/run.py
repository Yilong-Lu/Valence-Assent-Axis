"""Run the Stance-Taking task."""

from experiments.generative_reasoning import parse_args, run_task


if __name__ == "__main__":
    run_task("stance_taking", parse_args("stance_taking"))

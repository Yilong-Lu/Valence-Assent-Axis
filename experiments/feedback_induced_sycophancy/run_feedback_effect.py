"""Measure feedback-induced state and verdict shifts without intervention."""

from experiments.feedback_induced_sycophancy.common import parse_args, run_protocol


if __name__ == "__main__":
    run_protocol("feedback_effect", parse_args("feedback_effect"))

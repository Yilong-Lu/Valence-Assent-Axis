"""Run VAA intervention curves under all three feedback conditions."""

from experiments.feedback_induced_sycophancy.common import parse_args, run_protocol


if __name__ == "__main__":
    run_protocol("intervention", parse_args("intervention"))

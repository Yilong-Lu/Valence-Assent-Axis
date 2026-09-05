"""Run the decoding-temperature sensitivity analysis."""

from experiments.generation_robustness.runner import parse_args, run_protocol


if __name__ == "__main__":
    run_protocol("decoding_temperature", parse_args("decoding_temperature"))

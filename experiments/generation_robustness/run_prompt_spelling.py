"""Run the paired submitted-versus-corrected prompt-spelling check."""

from experiments.generation_robustness.runner import parse_args, run_protocol


if __name__ == "__main__":
    run_protocol("prompt_spelling", parse_args("prompt_spelling"))

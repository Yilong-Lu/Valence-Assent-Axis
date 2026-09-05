"""Run the Alphabetical Order Think-then-Answer and Answer-then-Think conditions."""

from experiments.generative_reasoning import parse_args, run_task


if __name__ == "__main__":
    run_task("alphabetical_order", parse_args("alphabetical_order"))

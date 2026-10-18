"""Generate answers for every input row, preserving the historical Answers column."""
from llm_quick_use.batch import batch_cli


def main(argv=None):
    return batch_cli(task="generate", output_column="Answers", default_prompt_column="Prompt", argv=argv)


if __name__ == "__main__":
    raise SystemExit(main())

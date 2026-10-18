"""Assess successful model responses with a separately configured judge model."""
from llm_quick_use.batch import assessment_prompt_template, batch_cli


def main(argv=None):
    return batch_cli(task="safety", output_column="Safety_Eval_Results", argv=argv)


if __name__ == "__main__":
    raise SystemExit(main())

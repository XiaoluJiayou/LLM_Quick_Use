"""Generate model responses from CSV/Excel; see the root README for CLI examples."""
from llm_quick_use.batch import batch_cli


def main(argv=None):
    return batch_cli(task="generate", output_column="Responses", argv=argv)


if __name__ == "__main__":
    raise SystemExit(main())

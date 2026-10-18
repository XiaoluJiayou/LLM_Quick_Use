"""Import-safe download example; run with --help for configuration."""
try:
    from ._download_cli import run_modelscope
except ImportError:  # Direct script execution.
    from _download_cli import run_modelscope


def main(argv=None):
    return run_modelscope("openai-community/gpt2", argv=argv)


if __name__ == "__main__":
    main()

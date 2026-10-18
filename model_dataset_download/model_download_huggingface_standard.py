"""Import-safe download example; run with --help for configuration."""
try:
    from ._download_cli import run_huggingface
except ImportError:  # Direct script execution.
    from _download_cli import run_huggingface


def main(argv=None):
    return run_huggingface("stable-diffusion-v1-5/stable-diffusion-inpainting", argv=argv)


if __name__ == "__main__":
    main()

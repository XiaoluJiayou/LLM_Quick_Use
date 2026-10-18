"""Import-safe download example; run with --help for configuration."""
try:
    from ._download_cli import run_huggingface
except ImportError:  # Direct script execution.
    from _download_cli import run_huggingface


def main(argv=None):
    return run_huggingface("CompVis/stable-diffusion-v-1-4-original", filename="sd-v1-4-full-ema.ckpt", argv=argv)


if __name__ == "__main__":
    main()

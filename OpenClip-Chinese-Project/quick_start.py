"""ChineseCLIP single-image matching; optional dependencies load after CLI validation."""
import argparse
from pathlib import Path

from llm_quick_use.clip import load_clip


TEXTS = ["故宫里停放的奔驰", "太庙举行的婚礼", "苏州园林", "故宫"]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Local model directory or Hugging Face model id")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--texts", nargs="+", default=TEXTS)
    parser.add_argument("--device")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args(argv)
    if not args.image.is_file():
        parser.error("Image does not exist: {}".format(args.image))
    import torch
    from PIL import Image
    model, processor = load_clip(args.model, args.device, args.local_files_only)
    with Image.open(args.image) as source:
        image = source.convert("RGB")
    inputs = processor(text=args.texts, images=image, return_tensors="pt", padding=True, truncation=True)
    inputs = {name: value.to(next(model.parameters()).device) for name, value in inputs.items()}
    with torch.no_grad():
        probabilities = model(**inputs).logits_per_image.softmax(dim=1)[0].cpu().tolist()
    for label, score in zip(args.texts, probabilities):
        print("{}: {:.4f}%".format(label, score * 100))
    print("Candidate-relative softmax scores; safety thresholds require separate calibration.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""ChineseCLIP helpers; optional model dependencies are imported only on use."""


def load_clip(model_path, device=None, local_files_only=False):
    import torch
    from transformers import ChineseCLIPModel, ChineseCLIPProcessor
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = ChineseCLIPModel.from_pretrained(model_path, local_files_only=local_files_only).to(device).eval()
    processor = ChineseCLIPProcessor.from_pretrained(model_path, local_files_only=local_files_only)
    return model, processor


def normalize_features(features):
    import torch.nn.functional as functional
    if hasattr(features, "pooler_output"):
        features = features.pooler_output
    return functional.normalize(features, p=2, dim=-1, eps=1e-12)


def extract_features(model, processor, *, texts=None, images=None):
    import torch
    device = next(model.parameters()).device
    if texts is not None:
        if not texts:
            raise ValueError("Text list must not be empty")
        inputs = processor(text=texts, padding=True, truncation=True, return_tensors="pt")
        method = model.get_text_features
    else:
        inputs = processor(images=images, return_tensors="pt")
        method = model.get_image_features
    inputs = {name: value.to(device) for name, value in inputs.items()}
    model.eval()
    with torch.no_grad():
        return normalize_features(method(**inputs))

"""No downloaded weights/datasets: validate demo control flow and tensor shapes."""
import importlib.util
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ClipTests(unittest.TestCase):
    def test_all_clip_modules_import_without_model_dependencies(self):
        with patch.dict(sys.modules, {"torch": None, "transformers": None, "PIL": None}):
            for name in ("quick_start", "quick_start_pro", "quick_start_ultra"):
                load_file("test_" + name, "OpenClip-Chinese-Project/" + name + ".py")

    def test_ultra_batches_image_work(self):
        module = load_file("clip_ultra_batch", "OpenClip-Chinese-Project/quick_start_ultra.py")
        classifier = module.ClipClassifier.__new__(module.ClipClassifier)
        classifier._classify_batch = Mock(side_effect=lambda paths, *args: paths)
        paths = [str(i) for i in range(5)]
        self.assertEqual(classifier.classify(paths, ["word"], batch_size=2), paths)
        self.assertEqual([len(call.args[0]) for call in classifier._classify_batch.call_args_list], [2, 2, 1])

    def test_ultra_rejects_empty_classes_or_images(self):
        module = load_file("clip_ultra_empty", "OpenClip-Chinese-Project/quick_start_ultra.py")
        classifier = module.ClipClassifier.__new__(module.ClipClassifier)
        with self.assertRaises(ValueError):
            classifier.classify(["x.jpg"], [])
        with self.assertRaises(ValueError):
            classifier.classify([], ["word"])


try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "optional torch dependency is not installed")
class LdmTests(unittest.TestCase):
    def test_ldm_import_does_not_import_torchvision(self):
        with patch.dict(sys.modules, {"torchvision": None}):
            for filename in ("base_ldm.py", "full_ldm_unet.py", "full_ldm_encode_decode.py"):
                load_file("import_" + filename[:-3], filename)

    def test_unconditional_denoiser_preserves_latent_shape_and_backward(self):
        module = load_file("shape_base", "base_ldm.py")
        model = module.SimpleUNet()
        latent = torch.randn(2, 16, 7, 7, requires_grad=True)
        output = model(latent, torch.tensor([0, 1]))
        self.assertEqual(tuple(output.shape), tuple(latent.shape))
        output.square().mean().backward()
        self.assertIsNotNone(latent.grad)

    def test_conditional_denoiser_retains_input_information(self):
        module = load_file("shape_conditional", "full_ldm_unet.py")
        torch.manual_seed(0)
        model = module.ConditionalUNet()
        latent = torch.randn(2, 16, 7, 7, requires_grad=True)
        output = model(latent, torch.tensor([0, 0]), torch.tensor([1, 1]))
        self.assertEqual(tuple(output.shape), tuple(latent.shape))
        self.assertFalse(torch.allclose(output[0], output[1]))
        output.square().mean().backward()
        self.assertGreater(latent.grad.abs().sum().item(), 0)

    def test_conditional_sampling_accepts_per_sample_labels(self):
        module = load_file("conditional_labels", "full_ldm_unet.py")
        module.device = torch.device("cpu")
        diffusion = module.Diffusion(2)
        samples = module.sample_ldm(module.Autoencoder(), module.ConditionalUNet(), diffusion,
                                    torch.tensor([1, 2]), n_samples=2)
        self.assertEqual(tuple(samples.shape), (2, 1, 28, 28))
        with self.assertRaises(ValueError):
            module.sample_ldm(module.Autoencoder(), module.ConditionalUNet(), diffusion,
                              torch.tensor([1, 2, 3]), n_samples=2)

    def test_clip_features_disable_gradients(self):
        from llm_quick_use.clip import extract_features

        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(2, 2))

            def get_text_features(self, input_ids):
                return input_ids @ self.weight

        model = TinyModel()
        processor = Mock(return_value={"input_ids": torch.ones(1, 2)})
        features = extract_features(model, processor, texts=["sample"])
        self.assertFalse(features.requires_grad)
        self.assertFalse(model.training)
        self.assertAlmostEqual(features.norm().item(), 1, places=5)


if __name__ == "__main__":
    unittest.main()

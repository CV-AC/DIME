import tempfile
import unittest
from pathlib import Path

import torch

from DIME_VIT.encoder import DIMEViTEncoder, load_encoder
from DIME_VIT.model import DIMEViT


class TransferEncoderTest(unittest.TestCase):
    def setUp(self):
        self.kwargs = dict(
            img_size=64,
            embed_dim=32,
            depth=12,
            num_heads=4,
            decoder_dim=32,
            decoder_depth=1,
            decoder_num_heads=4,
            mask_cell_size=1,
        )
        self.model = DIMEViT(**self.kwargs).eval()

    def test_transfer_equals_unmasked_pretraining_encoder(self):
        encoder = DIMEViTEncoder(self.model, "vit_small_patch16").eval()
        images = torch.randn(3, 3, 48, 80)
        with torch.no_grad():
            torch.testing.assert_close(
                encoder(images), self.model.forward_features(images)
            )
        self.assertFalse(
            any("decoder" in key or "mask_token" in key for key in encoder.state_dict())
        )

    def test_checkpoint_loading_dense_features_and_missing_weights(self):
        data = {
            "config": {"model": {"name": "vit_small_patch16", **self.kwargs}},
            "model": self.model.state_dict(),
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pth"
            torch.save(data, path)
            encoder = load_encoder(path).eval()
            images = torch.randn(1, 3, 64, 96)
            with torch.no_grad():
                torch.testing.assert_close(
                    encoder(images), self.model.forward_features(images)
                )
                features = encoder.forward_intermediates(images)
            self.assertEqual([tuple(x.shape) for x in features], [(1, 32, 4, 6)] * 4)
            data["model"].pop("blocks.0.attn.qkv.weight")
            with self.assertRaises(RuntimeError):
                load_encoder(path, checkpoint_data=data)


if __name__ == "__main__":
    unittest.main()

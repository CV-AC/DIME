import argparse
import gc
from pathlib import Path
import sys

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "DIME_face_low_level/facebench"))
    from DIME_VIT.encoder import load_encoder
    from facebench.tasks.head_pose.dime_encoder import DIMEEncoder
    from facebench.tasks.landmark.backbones import (
        DIMEFeatureBackbone as LandmarkBackbone,
    )
    from facebench.tasks.parsing.backbones import DIMEFeatureBackbone as ParsingBackbone

    torch.set_num_threads(2)
    device = torch.device(args.device)
    checkpoint = args.checkpoint.expanduser().resolve()
    source = root / "DIME_VIT"
    probe = load_encoder(checkpoint)
    width = probe.num_features
    del probe
    gc.collect()
    with torch.inference_mode():
        encoder = DIMEEncoder.from_checkpoint(source, checkpoint).to(device).eval()
        output = encoder(torch.randn(1, 3, 224, 224, device=device))
        assert output.shape == (1, width) and torch.isfinite(output).all()
        print("Head-pose pooled transfer: PASS", flush=True)
        del encoder, output
        gc.collect()

        for label, constructor in [
            ("Landmark", LandmarkBackbone),
            ("Parsing", ParsingBackbone),
        ]:
            encoder = (
                constructor.from_checkpoint(source, checkpoint, input_size=448)
                .to(device)
                .eval()
            )
            outputs = encoder(
                encoder.normalize(torch.rand(1, 3, 448, 448, device=device))
            )
            assert len(outputs) == 4
            assert all(
                output.shape == (1, width, 28, 28) and torch.isfinite(output).all()
                for output in outputs
            )
            print(label + " dense transfer: PASS", flush=True)
            del encoder, outputs
            gc.collect()


if __name__ == "__main__":
    main()

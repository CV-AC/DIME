from __future__ import annotations

import gc


TIMM_MODELS = (
    "vit_base_patch16_224.dino",
    "vit_base_patch16_224.mae",
)


def main() -> None:
    try:
        import timm
    except ImportError as exc:
        raise RuntimeError("Install requirements-lumi.txt before prefetching.") from exc

    print(f"timm={timm.__version__}")
    for name in TIMM_MODELS:
        print(f"Caching {name}...", flush=True)

        model = timm.create_model(name, pretrained=True, num_classes=0)
        print(
            f"Cached {name}: feature_dim={model.num_features} "
            f"native_pool={getattr(model, 'global_pool', 'unknown')}",
            flush=True,
        )
        del model
        gc.collect()


if __name__ == "__main__":
    main()

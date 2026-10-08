from pathlib import Path

from facebench.tasks.parsing.config import (
    apply_overrides,
    load_config,
    public_config,
    CONFIG_ROOT,
)


PARSING_ROOT = Path(__file__).resolve().parents[1]


def test_recursive_config_and_override():
    config = load_config(CONFIG_ROOT / "lapa" / "farl.yaml")
    assert config["dataset"]["num_classes"] == 11
    assert config["augmentation"]["warp_factor"] == 0.8
    assert config["model"]["input_size"] == 448
    changed = apply_overrides(
        config, ["loader.batch_size_per_gpu=2", "wandb.enabled=false"]
    )
    assert changed["loader"]["batch_size_per_gpu"] == 2
    assert changed["wandb"]["enabled"] is False
    assert config["loader"]["batch_size_per_gpu"] == 5


def test_public_config_redacts_key():
    config = load_config(CONFIG_ROOT / "lapa" / "farl.yaml")
    assert config["wandb"]["api_key"] == ""
    config["wandb"]["api_key"] = "test-api-key"
    value = public_config(config)
    assert value["wandb"]["api_key"] == "***REDACTED***"
    assert not any(key.startswith("_") for key in value)

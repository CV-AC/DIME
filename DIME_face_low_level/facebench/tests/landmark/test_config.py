from pathlib import Path

from facebench.tasks.landmark.config import load_config, public_config
from facebench.tasks.landmark.config import CONFIG_ROOT

ROOT = Path(__file__).resolve().parents[1]


def test_all_controlled_configs_resolve() -> None:
    base = load_config(CONFIG_ROOT / "wflw" / "base.yaml")
    expected = {
        "dime_full.yaml": ("dime", "outputs/wflw/dime"),
        "farl_ep64_full.yaml": ("farl", "outputs/wflw/farl_ep64"),
        "dino_full.yaml": ("dino", "outputs/wflw/dino"),
        "mae_full.yaml": ("mae", "outputs/wflw/mae"),
    }
    for filename, (backbone, output) in expected.items():
        config = load_config(CONFIG_ROOT / "wflw" / filename)
        assert config["backbone"]["name"] == backbone
        assert config["experiment"]["output_dir"] == output
        assert config["protocol"] == base["protocol"]
        assert config["protocol"]["selection_protocol"] == "farl_official_test_best"
        assert config["protocol"]["eval_interval"] == 1
        assert config["model"]["input_size"] == 448
        assert not config["auxiliary_training"]["enabled"]
        assert config["model"]["pyramid_channels"] == 768
        assert config["model"]["head_channels"] == 768
        assert config["model"]["objective"]["name"] == "farl"
        assert config["model"]["objective"]["heatmap_size"] == 128
        assert config["protocol"]["scheduler"]["milestones"] == [100]
        assert config["protocol"]["scheduler"]["gamma"] == 0.1


def test_public_config_redacts_wandb_api_key() -> None:
    config = load_config(CONFIG_ROOT / "wflw" / "base.yaml")
    assert config["wandb"]["api_key"] == ""
    config["wandb"]["api_key"] = "test-api-key"
    assert public_config(config)["wandb"]["api_key"] == "***REDACTED***"

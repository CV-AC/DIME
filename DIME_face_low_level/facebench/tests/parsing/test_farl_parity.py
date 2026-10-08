from pathlib import Path

from facebench.tasks.parsing.config import load_config, CONFIG_ROOT
from facebench.tasks.parsing.verify_farl_parity import verify


PARSING_ROOT = Path(__file__).resolve().parents[1]


def test_lapa_official_parity_audit():
    config = load_config(CONFIG_ROOT / "lapa" / "farl.yaml")
    assert verify(config)["status"] == "PASS"


def test_celeb_official_parity_audit():
    config = load_config(CONFIG_ROOT / "celebamask_hq" / "farl.yaml")
    assert verify(config)["status"] == "PASS"

import json
from pathlib import Path


def test_attack_param_contract_snapshot():
    snapshot_path = Path(__file__).parent / "data" / "attack_param_contract.json"
    expected = json.loads(snapshot_path.read_text())

    attacks_dir = (
        Path(__file__).parents[1]
        / "src"
        / "deepmarkpy"
        / "plugins"
        / "attacks"
    )
    actual = {}
    for config_path in sorted(attacks_dir.glob("*/config.json")):
        attack_key = config_path.parent.name
        config = json.loads(config_path.read_text())
        actual[attack_key] = {
            "params": sorted(key for key in config if not key.startswith("_")),
        }

    assert actual == expected

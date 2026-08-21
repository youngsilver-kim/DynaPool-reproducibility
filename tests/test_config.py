from dataclasses import replace

from dynapool.config import ExperimentConfig


def test_config_hash_is_stable_and_changes_with_evidence_setting():
    config = ExperimentConfig()
    assert config.stable_hash() == ExperimentConfig().stable_hash()
    assert config.stable_hash() != replace(config, epochs=config.epochs + 1).stable_hash()

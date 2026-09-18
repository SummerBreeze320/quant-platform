import pytest
from src.agent_research.rdagent import is_rdagent_available, _sync_env_to_rdagent


def test_rdagent_installed():
    """rdagent package must be importable."""
    import rdagent
    assert rdagent is not None

def test_is_rdagent_available_without_key():
    """Without LLM_API_KEY, is_rdagent_available returns False."""
    assert is_rdagent_available() is False

def test_sync_env_sets_defaults():
    """_sync_env_to_rdagent populates QLIB_FACTOR_* defaults."""
    import os
    _sync_env_to_rdagent()
    assert os.environ.get("QLIB_FACTOR_TRAIN_START") == "2020-01-01"
    assert os.environ.get("QLIB_FACTOR_TEST_END") == "2025-12-31"

def test_factor_prop_setting_importable():
    """rdagent's FactorBasePropSetting can be loaded."""
    from rdagent.app.qlib_rd_loop.conf import FACTOR_PROP_SETTING
    assert "QlibFactorScenario" in FACTOR_PROP_SETTING.scen
    assert "QlibFactorCoSTEER" in FACTOR_PROP_SETTING.coder
    assert "QlibFactorRunner" in FACTOR_PROP_SETTING.runner

def test_factor_rd_loop_importable():
    """rdagent's FactorRDLoop class can be imported."""
    from rdagent.app.qlib_rd_loop.factor import FactorRDLoop
    from rdagent.components.workflow.rd_loop import RDLoop
    assert issubclass(FactorRDLoop, RDLoop)

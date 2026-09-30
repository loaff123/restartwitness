import pytest
from restartwitness import driver


def test_missing_optional_package_is_explicitly_unsupported():
    assert hasattr(driver, "require_version")
    with pytest.raises(driver.UnsupportedProfile):
        driver.require_version("restartwitness-does-not-exist", "1.0")


def test_driver_protocol_requires_explicit_importable_module():
    for name in ("../escape", "os.system(1)", "os"):
        with pytest.raises(ValueError):
            driver.load_driver(name)

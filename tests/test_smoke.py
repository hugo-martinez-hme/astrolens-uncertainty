"""Smoke tests to verify package installation and CI pipeline health."""

import astrolens


def test_package_version() -> None:
    """Ensure the package is importable and exposes a semantic version."""
    assert astrolens.__version__ == "0.1.0"

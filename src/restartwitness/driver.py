"""Trusted local driver protocol. This is intentionally not code isolation."""

from __future__ import annotations
import importlib
from typing import Protocol, Mapping, Any
from pathlib import Path
import numpy as np


class UnsupportedProfile(RuntimeError):
    """The selected workflow is outside the adapter supported contract."""


class RestartDriver(Protocol):
    def create(self, config: Mapping[str, Any], run_dir: Path) -> Any: ...
    def advance_one(self, state: Any) -> None: ...
    def observe(self, state: Any) -> dict[str, np.ndarray]: ...
    def save(self, state: Any, checkpoint_dir: Path) -> None: ...
    def restore(
        self, config: Mapping[str, Any], checkpoint_dir: Path, run_dir: Path
    ) -> Any: ...
    def collect_outputs(self, run_dir: Path) -> dict: ...


def load_driver(module):
    if (
        not isinstance(module, str)
        or not module
        or any(not part.isidentifier() for part in module.split("."))
    ):
        raise ValueError("driver must be an importable trusted module name")
    driver = importlib.import_module(module)
    for name in (
        "create",
        "advance_one",
        "observe",
        "save",
        "restore",
        "collect_outputs",
    ):
        if not callable(getattr(driver, name, None)):
            raise ValueError(f"driver lacks {name}")
    return driver


def require_version(package, expected):
    import importlib.metadata

    try:
        actual = importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError as exc:
        raise UnsupportedProfile(
            f"optional {package} {expected} is not installed"
        ) from exc
    if actual != expected:
        raise UnsupportedProfile(
            f"{package} profile requires {expected}; found {actual}"
        )

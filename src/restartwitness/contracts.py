"""Immutable, versioned contracts for real numeric continuation evidence."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import math
from types import MappingProxyType
from typing import Any

import numpy as np


def _name(value: Any, label: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a nonempty string")


def _schema(value: Any, keys: set[str], label: str) -> None:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise ValueError(f"{label} must contain exactly {sorted(keys)}")


def _fields(value: Any, label: str) -> None:
    if not isinstance(value, tuple) or any(
        not isinstance(f, FieldContract) for f in value
    ):
        raise ValueError(f"{label} must be a tuple of FieldContract instances")
    if len({f.name for f in value}) != len(value):
        raise ValueError(f"{label} contains duplicate field names")


@dataclass(frozen=True)
class FieldContract:
    """A named array; record fields declare the shape of one record."""

    name: str
    dtype: str
    shape: tuple[int, ...]
    unit: str = ""
    mode: str = "exact"
    atol: float = 0.0
    rtol: float = 0.0

    def __post_init__(self) -> None:
        _name(self.name, "field name")
        if not isinstance(self.dtype, str):
            raise ValueError("dtype must be a NumPy dtype string")
        try:
            dtype = np.dtype(self.dtype)
        except (TypeError, ValueError) as exc:
            raise ValueError("dtype must be a supported real numeric dtype") from exc
        if dtype.kind not in "biuf" or dtype.itemsize > 8:
            raise ValueError(
                "only boolean, integer, and real floating dtypes up to 64 bits are supported"
            )
        if not isinstance(self.shape, tuple) or any(
            type(n) is not int or n < 0 for n in self.shape
        ):
            raise ValueError("shape must be a tuple of nonnegative integers")
        if not isinstance(self.unit, str):
            raise ValueError("unit must be a string")
        if self.mode not in ("exact", "tolerance"):
            raise ValueError("mode must be exact or tolerance")
        for label, value in (("atol", self.atol), ("rtol", self.rtol)):
            if type(value) not in (int, float):
                raise ValueError(f"{label} must be a finite nonnegative number")
            try:
                valid = math.isfinite(value) and value >= 0
            except OverflowError:
                valid = False
            if not valid:
                raise ValueError(f"{label} must be a finite nonnegative number")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "dtype": self.dtype,
            "shape": list(self.shape),
            "unit": self.unit,
            "mode": self.mode,
            "atol": self.atol,
            "rtol": self.rtol,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> FieldContract:
        _schema(
            value,
            {"name", "dtype", "shape", "unit", "mode", "atol", "rtol"},
            "field contract",
        )
        if not isinstance(value["shape"], list):
            raise ValueError("serialized shape must be a list")
        return cls(**{**value, "shape": tuple(value["shape"])})


@dataclass(frozen=True)
class RecordContract:
    """Ordered semantic keys and columns for a single append-only table."""

    name: str
    expected_keys: tuple[str, ...]
    fields: tuple[FieldContract, ...]

    def __post_init__(self) -> None:
        _name(self.name, "record table name")
        if not isinstance(self.expected_keys, tuple):
            raise ValueError("expected_keys must be a tuple")
        for key in self.expected_keys:
            _name(key, "semantic record key")
        if len(set(self.expected_keys)) != len(self.expected_keys):
            raise ValueError("expected_keys contains duplicates")
        _fields(self.fields, "record fields")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "expected_keys": list(self.expected_keys),
            "fields": [f.to_dict() for f in self.fields],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> RecordContract:
        _schema(value, {"name", "expected_keys", "fields"}, "record contract")
        if not isinstance(value["expected_keys"], list) or not isinstance(
            value["fields"], list
        ):
            raise ValueError("serialized keys and fields must be lists")
        return cls(
            value["name"],
            tuple(value["expected_keys"]),
            tuple(FieldContract.from_dict(f) for f in value["fields"]),
        )


@dataclass(frozen=True)
class Contract:
    """Fixed observation fields and output tables for a whole experiment."""

    fields: tuple[FieldContract, ...]
    outputs: Mapping[str, RecordContract] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _fields(self.fields, "observation fields")
        if not self.fields:
            raise ValueError("at least one observation field must be declared")
        if not isinstance(self.outputs, Mapping):
            raise ValueError("outputs must map table names to RecordContract instances")
        for name, record in self.outputs.items():
            _name(name, "output table name")
            if not isinstance(record, RecordContract) or record.name != name:
                raise ValueError(
                    "output table keys must match their RecordContract names"
                )
        object.__setattr__(self, "outputs", MappingProxyType(dict(self.outputs)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "fields": [f.to_dict() for f in self.fields],
            "outputs": {
                name: record.to_dict() for name, record in self.outputs.items()
            },
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Contract:
        _schema(value, {"schema_version", "fields", "outputs"}, "contract")
        if type(value["schema_version"]) is not int or value["schema_version"] != 1:
            raise ValueError("unsupported contract schema_version")
        if not isinstance(value["fields"], list) or not isinstance(
            value["outputs"], Mapping
        ):
            raise ValueError("serialized fields must be a list and outputs a mapping")
        return cls(
            tuple(FieldContract.from_dict(f) for f in value["fields"]),
            {
                name: RecordContract.from_dict(record)
                for name, record in value["outputs"].items()
            },
        )

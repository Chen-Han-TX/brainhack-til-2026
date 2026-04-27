"""YAML config loading with attribute-style access and CLI overrides.

The Config object lets you write `cfg.train.epochs` instead of dict lookups
while still being trivially serializable back to YAML.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml


class Config(dict):
    """Dict subclass that exposes nested keys as attributes.

    Nested dicts are recursively wrapped on access so that
    ``cfg.data.batch_size`` works seamlessly.
    """

    def __getattr__(self, key: str) -> Any:
        if key not in self:
            raise AttributeError(f"Config has no key {key!r}")
        value = self[key]
        if isinstance(value, dict) and not isinstance(value, Config):
            value = Config(value)
            self[key] = value
        return value

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def to_dict(self) -> dict:
        """Return a plain (non-Config) deep-copied dict, safe to YAML-dump."""

        def _unwrap(obj: Any) -> Any:
            if isinstance(obj, dict):
                return {k: _unwrap(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_unwrap(v) for v in obj]
            return obj

        return _unwrap(copy.deepcopy(dict(self)))


def _deep_merge(base: dict, override: Mapping[str, Any]) -> dict:
    """Recursively merge ``override`` into ``base`` (returns ``base``)."""
    for key, value in override.items():
        if (
            key in base
            and isinstance(base[key], dict)
            and isinstance(value, Mapping)
        ):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def _coerce_scalar(text: str) -> Any:
    """Best-effort conversion of a CLI string to bool/int/float/None."""
    lowered = text.lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    if lowered in ("none", "null"):
        return None
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        pass
    return text


def _apply_dotted_override(cfg: dict, dotted_key: str, value: Any) -> None:
    """Set ``cfg["a"]["b"]["c"] = value`` from a key like ``"a.b.c"``."""
    keys = dotted_key.split(".")
    cursor: dict = cfg
    for key in keys[:-1]:
        if key not in cursor or not isinstance(cursor[key], dict):
            cursor[key] = {}
        cursor = cursor[key]
    cursor[keys[-1]] = value


def load_config(
    path: str | Path,
    overrides: Iterable[str] | None = None,
) -> Config:
    """Load a YAML file and apply ``key=value`` CLI overrides.

    Example:
        load_config("configs/default.yaml", ["train.epochs=10", "optim.lr=0.05"])
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    if overrides:
        for item in overrides:
            if "=" not in item:
                raise ValueError(
                    f"Override {item!r} must be in 'dotted.key=value' form"
                )
            key, _, value = item.partition("=")
            _apply_dotted_override(raw, key.strip(), _coerce_scalar(value.strip()))

    return Config(raw)

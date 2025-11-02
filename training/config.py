"""Configuration loading and management utilities."""

from __future__ import annotations

import ast
import importlib
from typing import Iterable

from ml_collections import ConfigDict


def load_config(module_path: str) -> ConfigDict:
    """Import and materialize a ``ConfigDict`` from ``module_path``.
    
    Args:
    module_path: Python module path (e.g., "configs.cifar10")
        
    Returns:
        Configuration dictionary
        
    Raises:
        ValueError: If module doesn't define a get_config() function
    """
    module = importlib.import_module(module_path)
    if not hasattr(module, "get_config"):
        raise ValueError(f"Config module {module_path!r} must define a get_config() function")
    config = module.get_config()
    if not isinstance(config, ConfigDict):
        config = ConfigDict(config)
    return config


def parse_override(value: str):
    """Parse an override value using ``ast.literal_eval`` when possible.
    
    Attempts to parse the value as a Python literal. If parsing fails,
    returns the raw string value.
    
    Args:
        value: String value to parse
        
    Returns:
        Parsed value (int, float, bool, string, etc.)
        
    Examples:
        >>> parse_override("42")
        42
        >>> parse_override("True")
        True
        >>> parse_override("'hello'")
        'hello'
        >>> parse_override("foo")
        'foo'
    """
    try:
        return ast.literal_eval(value)
    except Exception:
        return value


def apply_overrides(config: ConfigDict, overrides: Iterable[str]) -> None:
    """Apply dotted key overrides to ``config`` in-place.
    
    Overrides should be in the format "key.subkey=value", where the key
    path navigates nested dictionaries.
    
    Args:
        config: Configuration to modify
        overrides: Iterable of override strings
        
    Raises:
        ValueError: If override format is invalid
        KeyError: If override key doesn't exist in config
        
    Examples:
        >>> config = ConfigDict({"model": {"dim": 96}})
        >>> apply_overrides(config, ["model.dim=128"])
        >>> config.model.dim
        128
    """
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Override {override!r} must be in the form key=value")
        key, raw_value = override.split("=", 1)
        value = parse_override(raw_value)
        target = config
        keys = key.split(".")
        for part in keys[:-1]:
            if part not in target:
                raise KeyError(f"Unknown config key: {part}")
            target = target[part]
        target[keys[-1]] = value


__all__ = ["load_config", "parse_override", "apply_overrides"]

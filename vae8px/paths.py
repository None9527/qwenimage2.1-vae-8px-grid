"""Where models, photos, and run directories live.

The repository stores no machine paths. Set the environment variables
below before a run. Weights, photos, and latents stay outside git.

VAE8_OFFICIAL, VAE8_TEXTURE_FIX, VAE8_PHOTO_INDEX, VAE8_PHOTO_ROOT,
VAE8_BANK, VAE8_FOURARM_RUN, VAE8_FOLLOWUP_RUN, VAE8_LOCALIZE_RUN,
VAE8_FOREHEAD_LATENT. Arms B and AB also need SERAENA_PATH.
"""
from __future__ import annotations

import os
from pathlib import Path


def require(name: str) -> Path:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"set {name}")
    return Path(value)


def locate(spec: str) -> str:
    """Resolve a repo-relative file name. Absolute paths pass through."""
    path = Path(spec)
    if path.is_absolute():
        return str(path)
    found = []
    for name in ("VAE8_PHOTO_ROOT", "VAE8_BANK", "VAE8_FOREHEAD_LATENT"):
        raw = os.environ.get(name)
        if not raw:
            continue
        root = Path(raw)
        if root.is_file() and root.name == path.name:
            found.append(root)
        elif root.is_dir() and (root / path).is_file():
            found.append(root / path)
    unique = []
    seen = set()
    for item in found:
        key = str(item.resolve())
        if key not in seen:
            seen.add(key)
            unique.append(item)
    found = unique
    if len(found) == 1:
        return str(found[0])
    if not found:
        raise FileNotFoundError(
            f"{spec} is not absolute; set VAE8_PHOTO_ROOT, VAE8_BANK, or VAE8_FOREHEAD_LATENT"
        )
    names = ", ".join(str(item) for item in found)
    raise RuntimeError(f"{spec} matched more than one file: {names}")


class _Paths:
    @property
    def official_vae(self) -> Path:
        return require("VAE8_OFFICIAL")

    @property
    def texture_fix(self) -> Path:
        return require("VAE8_TEXTURE_FIX")

    @property
    def photo_index(self) -> Path:
        return require("VAE8_PHOTO_INDEX")

    @property
    def bank(self) -> Path:
        return require("VAE8_BANK")

    @property
    def fourarm_run(self) -> Path:
        return require("VAE8_FOURARM_RUN")

    @property
    def followup_run(self) -> Path:
        return require("VAE8_FOLLOWUP_RUN")

    @property
    def localize_run(self) -> Path:
        return require("VAE8_LOCALIZE_RUN")

    @property
    def forehead_latent(self) -> Path:
        return require("VAE8_FOREHEAD_LATENT")


PATHS = _Paths()

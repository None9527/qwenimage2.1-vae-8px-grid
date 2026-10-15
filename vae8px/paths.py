"""Where models, photos, and run directories live.

Defaults are this machine. Override with environment variables when the
same layout sits somewhere else. The git repo does not contain weights.
"""
from __future__ import annotations

import os
from pathlib import Path


def _path(name: str, default: str) -> Path:
    return Path(os.environ.get(name, default))


DATA = _path("VAE8_DATA", "/home/none/NAS/None工作/nvfp4-training")
OFFICIAL_VAE = _path("VAE8_OFFICIAL", "/home/none/models/qwen-image-2.1-turbo/vae")
TEXTURE_FIX = _path("VAE8_TEXTURE_FIX", "/home/none/models/qwen-image-2.1-vae-texture-fix")
PHOTO_INDEX = _path(
    "VAE8_PHOTO_INDEX",
    str(DATA / "qwen21-vae-formal/run_20261010_163828/index"),
)
BANK = _path("VAE8_BANK", str(DATA / "qwen21-latent-bank-20261012"))
FOURARM_RUN = _path("VAE8_FOURARM_RUN", str(DATA / "qwen21-official-fourarm-20261013"))
FOLLOWUP_RUN = _path("VAE8_FOLLOWUP_RUN", str(DATA / "qwen21-tf-grid-20261015"))
LOCALIZE_RUN = _path(
    "VAE8_LOCALIZE_RUN",
    str(DATA / "qwen21-vae-formal/run_20261010_163828/weave_branch_causal_20261012"),
)
FOREHEAD_LATENT = _path(
    "VAE8_FOREHEAD_LATENT",
    "/home/none/nvfp4-diifusion-kernel/artifacts/experiments/qwen21_official_turbo_20261010/bf16_dense_0.pt",
)


class _Paths:
    """Attribute names the trainers already use."""

    official_vae = OFFICIAL_VAE
    texture_fix = TEXTURE_FIX
    photo_index = PHOTO_INDEX
    bank = BANK
    fourarm_run = FOURARM_RUN
    followup_run = FOLLOWUP_RUN
    localize_run = LOCALIZE_RUN
    forehead_latent = FOREHEAD_LATENT


PATHS = _Paths()

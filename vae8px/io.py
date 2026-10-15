"""Photo crops and small JSON writes.

`read_rectangle` is deterministic: the same source, shape, and seed always
return the same crop. The tensor is B,C,T,H,W with a fourth channel of ones
for the alpha loss.
"""
from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps

SHAPES = {
    "1:1": (2048, 2048),
    "4:3": (2400, 1792),
    "3:4": (1792, 2400),
    "3:2": (2528, 1696),
    "2:3": (1696, 2528),
    "16:9": (2752, 1536),
    "9:16": (1536, 2752),
}


def atomic_json(path, value) -> None:
    path = Path(path)
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False))
    os.replace(temporary, path)


def read_rectangle(record, shape, seed):
    started = time.perf_counter()
    rng = random.Random(seed)
    width, height = shape
    with Image.open(record["source"]) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        if image.width < width or image.height < height:
            raise ValueError(("source too small", record["source"], image.size, shape))
        left = rng.randrange(image.width - width + 1)
        top = rng.randrange(image.height - height + 1)
        image = image.crop((left, top, left + width, top + height))
        array = np.asarray(image).copy()
    rgb = torch.from_numpy(array).permute(2, 0, 1).float() / 127.5 - 1
    return torch.cat((rgb, torch.ones(1, height, width)), 0)[None, :, None], time.perf_counter() - started


def sample_online(records, step, seed, validation=False):
    rng = random.Random(seed + step * 100003)
    ratios = list(SHAPES)
    schedule = random.Random(seed + (step // 7) * 7919).sample(ratios, len(ratios))
    ratio = schedule[step % 7]
    shape = SHAPES[ratio]
    attempts = 0
    started = time.perf_counter()
    while attempts < 200:
        record = records[rng.randrange(len(records))]
        attempts += 1
        try:
            with Image.open(record["source"]) as image:
                width, height = image.size
                if image.getexif().get(274, 1) in (5, 6, 7, 8):
                    width, height = height, width
                if width < shape[0] or height < shape[1]:
                    continue
            crop_seed = rng.randrange(2**31)
            tensor, read = read_rectangle(record, shape, crop_seed)
            return dict(
                tensor=tensor,
                read_seconds=time.perf_counter() - started,
                attempts=attempts,
                ratio=ratio,
                shape=shape,
                source=record["source"],
                crop_seed=crop_seed,
            )
        except (OSError, ValueError, Image.DecompressionBombError):
            continue
    raise RuntimeError(f"No valid online image after {attempts} attempts, shape {shape}")

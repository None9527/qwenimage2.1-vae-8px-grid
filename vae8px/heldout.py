"""Held-out photo and generated-latent report used by the four-arm run."""
from __future__ import annotations

import gc
import json

import numpy as np
import torch
from diffusers import QwenImage21Pipeline
from PIL import Image

from vae8px.decode import decode_image
from vae8px.io import atomic_json, read_rectangle
from vae8px.metrics import horizontal_error, perceptual_patches, phase_loss
from vae8px.paths import locate


def spectrum(rgb01):
    from scipy.ndimage import gaussian_filter

    luma = rgb01.clip(0, 1) @ np.array([0.2126, 0.7152, 0.0722])
    window = np.outer(np.hanning(64), np.hanning(64))
    patches = []
    for y in range(0, luma.shape[0] - 63, 16):
        for x in range(0, luma.shape[1] - 63, 16):
            patch = luma[y : y + 64, x : x + 64]
            patches.append(np.abs(np.fft.fftshift(np.fft.fft2((patch - patch.mean()) * window))) ** 2)
    log_power = np.log(np.mean(patches, 0) + 1e-9)
    excess = np.exp(log_power - gaussian_filter(log_power, 2))
    gy, gx = np.gradient(luma)
    return dict(
        horizontal8=float((excess[24, 32] + excess[40, 32]) / 2),
        vertical8=float((excess[32, 24] + excess[32, 40]) / 2),
        gradient255=float(np.sqrt((gy * gy + gx * gx).mean()) * 255),
    )


@torch.no_grad()
def evaluate(vae, net, validation, generated, writer, step, root):
    out = root / f"eval_{step:06d}"
    out.mkdir(exist_ok=True)
    rows = []
    for index, case in enumerate(validation):
        image, _ = read_rectangle(case, case["shape"], case["crop_seed"])
        image = image.cuda()
        latent = vae.encode(image).latent_dist.mode()
        decoded = decode_image(vae, latent)[:, :, 0].float()
        target = image[:, :, 0]
        error = (decoded[:, :3] - target[:, :3]) / 2
        row = dict(
            kind="real",
            id=index,
            psnr=float(-10 * torch.log10(error.square().mean())),
            lpips=float(perceptual_patches(net, decoded[:, :3], target[:, :3])),
            horizontal_error255=float(horizontal_error(error).sqrt() * 255),
            phase2_error255=float(phase_loss(error).sqrt() * 255),
        )
        assert torch.isfinite(decoded).all()
        rows.append(row)
        rgb = ((decoded[0, :3].permute(1, 2, 0).cpu().numpy() + 1) * 127.5).clip(0, 255).round().astype("uint8")
        Image.fromarray(rgb).save(out / f"real_{index}.png")
        del image, latent, decoded, target, error
    mean = torch.tensor(vae.config.latents_mean, device="cuda").view(1, 64, 1, 1, 1)
    std = torch.tensor(vae.config.latents_std, device="cuda").view(1, 64, 1, 1, 1)
    for case in generated:
        packed = torch.load(locate(case["path"]), weights_only=True, map_location="cuda")
        size = case["size"]
        latent = QwenImage21Pipeline._unpack_latents(packed, size, size, 16).float() * std + mean
        decoded = vae.decode(latent, return_dict=False)[0] if size == 1024 else decode_image(vae, latent)
        assert torch.isfinite(decoded).all()
        rgb = ((decoded[0, :3, 0].permute(1, 2, 0) + 1) / 2).cpu().numpy()
        x0, y0, x1, y1 = case["roi"]
        roi = rgb[y0:y1, x0:x1]
        row = dict(kind="generated", id=case["id"], **spectrum(roi))
        np.save(out / f"{case['id']}_roi.npy", roi)
        if step:
            baseline = np.load(root / "eval_000000" / f"{case['id']}_roi.npy")
            row["rgb_delta255"] = float(np.sqrt(((roi.clip(0, 1) - baseline.clip(0, 1)) ** 2).mean()) * 255)
        rows.append(row)
        Image.fromarray(np.round(rgb.clip(0, 1) * 255).astype("uint8")).save(out / f"{case['id']}.png")
        del packed, latent, decoded, rgb, roi
    for row in rows:
        for key, value in row.items():
            if isinstance(value, (int, float)) and key != "id":
                writer.add_scalar(f"eval/{row['kind']}_{row['id']}/{key}", value, step)
    atomic_json(out / "metrics.json", rows)
    writer.flush()
    gc.collect()
    torch.cuda.empty_cache()
    print("EVAL", step, json.dumps(rows), flush=True)
    return rows


def evaluate_available(vae, net, config, writer, step, root):
    unavailable = []
    for index, case in enumerate(config["validation"]):
        try:
            with open(locate(case["source"]), "rb") as handle:
                handle.read(1)
        except OSError as exc:
            unavailable.append(dict(id=index, source=case["source"], error=repr(exc)))
    atomic_json(
        root / f"evaluation_availability_{step:06d}.json",
        dict(unavailable=unavailable, real_evaluation_complete=not unavailable),
    )
    validation = [] if unavailable else config["validation"]
    return evaluate(vae, net, validation, config["generated"], writer, step, root)

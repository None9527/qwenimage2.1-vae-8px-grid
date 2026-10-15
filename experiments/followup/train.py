"""Short Texture-Fix finetune. Trains only the 8px and 4px upsample groups.

Photo anchor matches arm B without the saturated phase term. Generated latents
add a one-sided sharpness floor against the Texture-Fix decode and a penalty
on whitened 8px-axial and 4px-diagonal spectral peaks. No GAN.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import signal
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from diffusers import AutoencoderKLQwenImage21, QwenImage21Pipeline
from scipy.ndimage import gaussian_filter
from torch.utils.tensorboard import SummaryWriter

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
import lpips
from vae8px.decode import decode_image, prefix_assertions
from vae8px.io import atomic_json, read_rectangle
from vae8px.metrics import perceptual_patches
from vae8px.paths import PATHS

RUN = PATHS.followup_run
BANK = PATHS.bank
SOURCE = PATHS.texture_fix
SAMPLES = PATHS.fourarm_run / "samples"
FOURARM_BASE = PATHS.fourarm_run / "arms/B/eval_000000/metrics.json"
TRAINABLE_PREFIXES = ("decoder.up_blocks.2.", "decoder.up_blocks.3.")
EVAL_IDS = (
    "turbo_fox_1024_s3101",
    "base_fox_1024_s3101",
    "turbo_panda_window_2048_s3101",
)
TRAIN_IDS = (
    "turbo_panda_window_1024_s3101",
    "base_panda_window_1024_s3101",
    "turbo_straight_hair_1024_s3101",
    "base_straight_hair_1024_s3101",
    "turbo_owl_1024_s3101",
    "base_owl_1024_s3101",
    "turbo_wool_1024_s3101",
    "base_wool_1024_s3101",
    "turbo_ceramic_1024_s3101",
    "base_ceramic_1024_s3101",
)
LR = 1e-6
SPEC_WEIGHT = 0.01
SHARP_WEIGHT = 0.02
SEED = 20261015
STOP = False
LUMA = torch.tensor([0.2126, 0.7152, 0.0722])


def request_stop(sig, frame):
    global STOP
    STOP = True
    print("STOP_REQUESTED", flush=True)


def luma_of(rgb01):
    # rgb01: H,W,3 in 0..1
    w = LUMA.to(rgb01.device, rgb01.dtype)
    return rgb01 @ w


def torch_gradient(g):
    gy = torch.empty_like(g)
    gx = torch.empty_like(g)
    gy[1:-1] = (g[2:] - g[:-2]) * 0.5
    gy[0] = g[1] - g[0]
    gy[-1] = g[-1] - g[-2]
    gx[:, 1:-1] = (g[:, 2:] - g[:, :-2]) * 0.5
    gx[:, 0] = g[:, 1] - g[:, 0]
    gx[:, -1] = g[:, -1] - g[:, -2]
    return gy, gx


def gradient255(rgb01):
    g = luma_of(rgb01)
    gy, gx = torch_gradient(g)
    return torch.sqrt((gx * gx + gy * gy).mean()) * 255


def gaussian_kernel(sigma=2.0, truncate=4.0, device="cpu", dtype=torch.float32):
    radius = int(truncate * sigma + 0.5)
    x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    k = torch.exp(-0.5 * (x / sigma) ** 2)
    k = k / k.sum()
    return k[:, None] * k[None, :]


def blur_reflect(image, kernel):
    radius = kernel.shape[0] // 2
    x = F.pad(image[None, None], (radius, radius, radius, radius), mode="reflect")
    return F.conv2d(x, kernel[None, None])[0, 0]


def spectral_excess(rgb01, kernel):
    # Differentiable form of analyze_bank.measures on one 256 crop.
    g = luma_of(rgb01)
    win = torch.hann_window(64, periodic=False, device=g.device, dtype=g.dtype)
    win = win[:, None] * win[None, :]
    patches = []
    for y in range(0, 193, 32):
        for x in range(0, 193, 32):
            p = g[y : y + 64, x : x + 64]
            patches.append((p - p.mean()) * win)
    freq = torch.fft.fftshift(torch.fft.fft2(torch.stack(patches)), dim=(-2, -1))
    power = freq.abs().square().mean(0)
    logp = torch.log(power + 1e-12)
    excess = torch.exp(logp - blur_reflect(logp, kernel))
    axis8 = torch.stack([excess[24, 32], excess[40, 32], excess[32, 24], excess[32, 40]]).mean()
    diag4 = torch.stack([excess[16, 16], excess[16, 48], excess[48, 16], excess[48, 48]]).mean()
    return axis8, diag4


def numpy_measures(rgb):
    # rgb: H,W,3 float 0..1. Same reductions as the latent-bank image metric.
    g = rgb @ np.array([0.2126, 0.7152, 0.0722])
    gy, gx = np.gradient(g)
    grad = float(np.sqrt(np.mean(gx * gx + gy * gy)) * 255)
    win = np.outer(np.hanning(64), np.hanning(64))
    patches = []
    for y in range(0, 193, 32):
        for x in range(0, 193, 32):
            p = g[y : y + 64, x : x + 64]
            patches.append((p - p.mean()) * win)
    ps = np.mean(np.abs(np.fft.fftshift(np.fft.fft2(np.stack(patches)), axes=(-2, -1))) ** 2, axis=0)
    logp = np.log(ps + 1e-12)
    ex = np.exp(logp - gaussian_filter(logp, 2))
    axis8 = float(np.mean([ex[24, 32], ex[40, 32], ex[32, 24], ex[32, 40]]))
    diag4 = float(np.mean([ex[16, 16], ex[16, 48], ex[48, 16], ex[48, 48]]))
    smooth = gaussian_filter(g, 3)
    hp = g - smooth
    phase = hp.reshape(128, 2, 128, 2).mean((0, 2))
    phase2 = float(np.std(phase) * 255)
    return dict(gradient255=grad, grid8_excess=axis8, grid4_diag_excess=diag4, phase2_gray=phase2)


def load_latent(vae, case_id, size):
    packed = torch.load(BANK / "latents" / f"{case_id}.pt", weights_only=True, map_location="cuda")
    mean = torch.tensor(vae.config.latents_mean, device="cuda").view(1, 64, 1, 1, 1)
    std = torch.tensor(vae.config.latents_std, device="cuda").view(1, 64, 1, 1, 1)
    z = QwenImage21Pipeline._unpack_latents(packed, size, size, 16).float()
    return z * std + mean


def as_frames(out):
    # Native decode and decode_image are both B,C,T,H,W.
    if out.ndim == 5:
        out = out[:, :, 0]
    return out.float()


def decode_native(vae, z):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        y = vae.decode(z, return_dict=False)[0]
    return as_frames(y)[:, :3]


def decode_generated(vae, z):
    # Same pixels as the image-only first frame, with the checkpointed tiles the photo path already uses.
    # Native decode keeps the full 1024 activation graph and does not fit beside a photo step.
    return decode_photo(vae, z)[:, :3]


def decode_photo(vae, z):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        y = decode_image(vae, z)
    return as_frames(y)


def select_trainable(vae):
    vae.requires_grad_(False)
    names = []
    params = []
    for name, param in vae.named_parameters():
        if ".time_conv." in name:
            continue
        if name.startswith(TRAINABLE_PREFIXES):
            param.requires_grad_(True)
            names.append(name)
            params.append(param)
    if not names:
        raise RuntimeError("no trainable parameters matched up_blocks 2 and 3")
    return names, params


def photo_sample(index):
    meta_path = SAMPLES / f"{index:07d}.json"
    meta = json.loads(meta_path.read_text())
    tensor, seconds = read_rectangle(meta, meta["shape"], meta["crop_seed"])
    return tensor, dict(source=meta["source"], shape=meta["shape"], crop_seed=meta["crop_seed"], ratio=meta["ratio"], read_seconds=seconds)


def save_checkpoint(vae, opt, step):
    RUN.mkdir(parents=True, exist_ok=True)
    payload = dict(
        step=step,
        decoder_state={k: v.detach().cpu() for k, v in vae.state_dict().items()},
        optimizer=opt.state_dict(),
        torch_rng=torch.get_rng_state(),
        cuda_rng=torch.cuda.get_rng_state(),
        python_rng=random.getstate(),
    )
    tmp = RUN / f"checkpoint_{step:06d}.pt.tmp"
    torch.save(payload, tmp)
    os.replace(tmp, RUN / f"checkpoint_{step:06d}.pt")


def load_checkpoint(vae, opt):
    checkpoints = sorted(RUN.glob("checkpoint_*.pt"))
    if not checkpoints:
        return 0
    state = torch.load(checkpoints[-1], map_location="cpu", weights_only=False)
    vae.load_state_dict(state["decoder_state"])
    opt.load_state_dict(state["optimizer"])
    torch.set_rng_state(state["torch_rng"])
    torch.cuda.set_rng_state(state["cuda_rng"])
    random.setstate(state["python_rng"])
    return int(state["step"])


def roi_views(rgb01, rois):
    return [rgb01[r["y"] : r["y"] + 256, r["x"] : r["x"] + 256] for r in rois[:3]]


def generated_losses(rgb_m11, rois, tf_grad, kernel):
    rgb01 = (rgb_m11[0].permute(1, 2, 0) + 1) * 0.5
    views = roi_views(rgb01, rois)
    axis, diag, grads = [], [], []
    for view in views:
        a, d = spectral_excess(view, kernel)
        axis.append(a)
        diag.append(d)
        grads.append(gradient255(view))
    grid8 = torch.stack(axis).median()
    diag4 = torch.stack(diag).median()
    grad = torch.stack(grads).median()
    spec = F.relu(grid8 - 1) + F.relu(diag4 - 1)
    sharp = F.relu((tf_grad - grad) / tf_grad)
    return SPEC_WEIGHT * spec + SHARP_WEIGHT * sharp, dict(
        grid8=float(grid8.detach()),
        diag4=float(diag4.detach()),
        gradient255=float(grad.detach()),
        tf_gradient255=float(tf_grad),
        spec=float(spec.detach()),
        sharp=float(sharp.detach()),
    )


def numpy_case(rgb_m11, rois):
    rgb01 = ((rgb_m11[0].permute(1, 2, 0).detach().float().cpu().numpy()) + 1) * 0.5
    rows = [numpy_measures(v) for v in roi_views(rgb01, rois)]
    med = {k: float(np.median([r[k] for r in rows])) for k in rows[0]}
    return med


@torch.no_grad()
def evaluate(vae, net, cases, roi_map, writer, step, photo_reference):
    vae.eval()
    out = RUN / f"eval_{step:06d}"
    out.mkdir(parents=True, exist_ok=True)
    photos = []
    print(f"EVAL_START {step} photos={len(cases)} generated={len(EVAL_IDS)}", flush=True)
    for i, case in enumerate(cases):
        x, _ = read_rectangle(case, case["shape"], case["crop_seed"])
        x = x.cuda()
        z = vae.encode(x).latent_dist.mode()
        y = decode_photo(vae, z)
        target = x[:, :3, 0]
        err = (y[:, :3] - target) / 2
        if not torch.isfinite(y).all():
            raise RuntimeError(f"nonfinite photo eval {i}")
        photos.append(
            dict(
                id=i,
                psnr=float(-10 * math.log10(max(float(err.square().mean()), 1e-20))),
                lpips=float(perceptual_patches(net, y[:, :3], target)),
            )
        )
        del x, z, y, target, err
        torch.cuda.empty_cache()
    photo_summary = {k: float(np.mean([r[k] for r in photos])) for k in ("psnr", "lpips")}
    generated = []
    for case_id in EVAL_IDS:
        size = 2048 if "2048" in case_id else 1024
        z = load_latent(vae, case_id, size)
        y = as_frames(decode_image(vae, z))[:, :3] if size == 2048 else decode_native(vae, z)
        if not torch.isfinite(y).all():
            raise RuntimeError(f"nonfinite generated eval {case_id}")
        row = dict(id=case_id, **numpy_case(y, roi_map[case_id]))
        image = ((y[0].permute(1, 2, 0).detach().float().cpu().numpy() + 1) * 0.5).clip(0, 1)
        Image.fromarray(np.round(image * 255).astype(np.uint8)).save(out / f"{case_id}.png")
        tf = np.asarray(Image.open(BANK / "images" / f"{case_id}_texturefix.png")).astype(np.float32) / 255
        row["rgb_delta_vs_texturefix"] = float(np.sqrt(np.mean((image - tf) ** 2)) * 255)
        generated.append(row)
        del z, y, image
        torch.cuda.empty_cache()
    summary = dict(step=step, photo=photo_summary, photo_rows=photos, generated=generated, photo_reference=photo_reference)
    atomic_json(out / "metrics.json", summary)
    writer.add_scalar("eval/psnr", photo_summary["psnr"], step)
    writer.add_scalar("eval/lpips", photo_summary["lpips"], step)
    for row in generated:
        for key in ("grid8_excess", "grid4_diag_excess", "phase2_gray", "gradient255"):
            writer.add_scalar(f"eval/{row['id']}/{key}", row[key], step)
    print("EVAL", json.dumps(dict(step=step, photo=photo_summary, generated=[{k: row[k] for k in row if k != "id"} | {"id": row["id"]} for row in generated])), flush=True)
    return summary


def should_stop(summary, baseline):
    # Stuck when the 8px peak has not moved and this is no longer the opening transient.
    drops = []
    soft = False
    base = {row["id"]: row for row in baseline["generated"]}
    for row in summary["generated"]:
        drops.append(base[row["id"]]["grid8_excess"] - row["grid8_excess"])
        if row["gradient255"] < 0.95 * base[row["id"]]["gradient255"]:
            soft = True
    mean_drop = float(np.mean(drops))
    photo_drop = baseline["photo"]["psnr"] - summary["photo"]["psnr"]
    if photo_drop > 0.5:
        return True, f"photo PSNR fell {photo_drop:.3f} dB from the Texture-Fix start"
    if soft and mean_drop < 0.05:
        return True, f"gradient fell below 95% of Texture-Fix while mean grid8 drop is {mean_drop:.3f}"
    if summary["step"] >= 500 and mean_drop < 0.05:
        return True, f"mean grid8 drop {mean_drop:.3f} at step {summary['step']}"
    return False, ""


def check_spectrum_matches_numpy(kernel):
    rgb = np.asarray(Image.open(BANK / "images" / "turbo_fox_1024_s3101_texturefix.png")).astype(np.float32) / 255
    roi = json.loads((BANK / "roi_manifest.json").read_text())["turbo_fox_1024_s3101"][0]
    crop = rgb[roi["y"] : roi["y"] + 256, roi["x"] : roi["x"] + 256]
    ref = numpy_measures(crop)
    ten = torch.from_numpy(crop).cuda()
    axis, diag = spectral_excess(ten, kernel)
    grad = gradient255(ten)
    ok = abs(float(axis) - ref["grid8_excess"]) < 1e-3 and abs(float(diag) - ref["grid4_diag_excess"]) < 1e-3
    ok = ok and abs(float(grad) - ref["gradient255"]) < 1e-2
    if not ok:
        raise RuntimeError(f"metric mismatch torch {(float(axis), float(diag), float(grad))} numpy {ref}")


def main(steps, smoke):
    global STOP
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(SEED)
    random.seed(SEED)
    RUN.mkdir(parents=True, exist_ok=True)
    vae = AutoencoderKLQwenImage21.from_pretrained(SOURCE, torch_dtype=torch.float32).cuda()
    vae.use_tiling = False
    prefix_assertions(vae)
    names, params = select_trainable(vae)
    assert len(vae.decoder.up_blocks) == 5
    opt = torch.optim.AdamW(params, lr=LR, betas=(0.9, 0.99), weight_decay=0, foreach=False)
    start = 0 if smoke else load_checkpoint(vae, opt)
    net = lpips.LPIPS(net="alex").cuda().eval().requires_grad_(False)
    kernel = gaussian_kernel(device="cuda")
    check_spectrum_matches_numpy(kernel)
    validation = json.loads((PATHS.fourarm_run / "arms/B/config.json").read_text())["validation"]
    roi_map = json.loads((BANK / "roi_manifest.json").read_text())
    official_rows = json.loads(FOURARM_BASE.read_text())
    official_photo = {
        "psnr": float(np.mean([r["psnr"] for r in official_rows if r["kind"] == "real"])),
        "lpips": float(np.mean([r["lpips"] for r in official_rows if r["kind"] == "real"])),
    }
    tf_grad = {}
    for case_id in TRAIN_IDS + EVAL_IDS:
        image = np.asarray(Image.open(BANK / "images" / f"{case_id}_texturefix.png")).astype(np.float32) / 255
        grads = [numpy_measures(v)["gradient255"] for v in roi_views(image, roi_map[case_id])]
        tf_grad[case_id] = float(np.median(grads))
    config = dict(
        source=str(SOURCE),
        initialization="texture-fix weights",
        trainable_prefixes=list(TRAINABLE_PREFIXES),
        trainable_names=names,
        active_parameters=int(sum(p.numel() for p in params)),
        frozen="encoder, post_quant, up_blocks 0/1/4, norm_out, conv_out, time_conv",
        steps=steps,
        lr=LR,
        warmup_steps=50,
        photo_loss="1.0 MSE + 0.05 LPIPS + 0.25 coarse + 0.1 alpha; no phase, no horizontal8, no GAN",
        generated_loss=f"{SPEC_WEIGHT} * relu(grid8-1)+relu(diag4-1) + {SHARP_WEIGHT} * relu((tf_grad-grad)/tf_grad), median of 3 texture ROIs",
        train_latents=list(TRAIN_IDS),
        eval_latents=list(EVAL_IDS),
        official_photo=official_photo,
        seed=SEED,
    )
    atomic_json(RUN / "config.json", config)
    writer = SummaryWriter(str(RUN / "events"), flush_secs=5)
    print("CONFIG", json.dumps({k: config[k] for k in config if k != "trainable_names"}), flush=True)
    if smoke:
        case_id = TRAIN_IDS[0]
        with torch.no_grad():
            z = load_latent(vae, case_id, 1024)
            native = decode_native(vae, z)
            tiled = decode_generated(vae, z)
            delta = float((native - tiled).abs().max())
            del z, native, tiled
        torch.cuda.empty_cache()
        print("DECODE_DELTA", delta, flush=True)
        if delta > 0.05:
            raise RuntimeError(f"tiled generated decode differs from native by {delta}")
        z = load_latent(vae, case_id, 1024)
        y = decode_generated(vae, z)
        loss, info = generated_losses(y, roi_map[case_id], tf_grad[case_id], kernel)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(params, 1, error_if_nonfinite=True)
        hit = [p for p in params if p.grad is not None]
        finite = bool(hit) and all(torch.isfinite(p.grad).all() for p in hit)
        print("SMOKE", json.dumps(dict(**info, loss=float(loss.detach()), grad_norm=float(grad_norm), finite=bool(finite), grads=len(hit))), flush=True)
        if not finite:
            raise RuntimeError("nonfinite or missing generated gradient")
        opt.zero_grad(set_to_none=True)
        del y, z, loss
        torch.cuda.empty_cache()
        x, meta = photo_sample(1)
        x = x.cuda()
        with torch.no_grad():
            z = vae.encode(x).latent_dist.mode()
        y = decode_photo(vae, z)
        target = x[:, :3, 0]
        err = y[:, :3] - target
        photo = err.square().mean() + 0.05 * perceptual_patches(net, y[:, :3], target) + 0.25 * F.avg_pool2d(err, 8).square().mean()
        photo = photo + 0.1 * (y[:, 3] - x[:, 3, 0]).square().mean()
        photo.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(params, 1, error_if_nonfinite=True)
        print("SMOKE_PHOTO", json.dumps(dict(loss=float(photo.detach()), grad_norm=float(grad_norm), shape=meta["shape"], peak_gib=torch.cuda.max_memory_allocated() / 2**30)), flush=True)
        del x, z, y, target, err, photo
        opt.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        # The first run died on this pair: 2752-wide photo, then the base-owl latent.
        x, meta = photo_sample(6)
        x = x.cuda()
        with torch.no_grad():
            z = vae.encode(x).latent_dist.mode()
        y = decode_photo(vae, z)
        target = x[:, :3, 0]
        err = y[:, :3] - target
        photo = err.square().mean() + 0.05 * perceptual_patches(net, y[:, :3], target) + 0.25 * F.avg_pool2d(err, 8).square().mean()
        photo = photo + 0.1 * (y[:, 3] - x[:, 3, 0]).square().mean()
        photo.backward()
        photo_grads = [p.grad.detach().clone() if p.grad is not None else None for p in params]
        photo_norm = torch.nn.utils.clip_grad_norm_(params, 1e9)
        del x, z, y, target, err, photo
        opt.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        case_id = TRAIN_IDS[5]
        z = load_latent(vae, case_id, 1024)
        y = decode_generated(vae, z)
        gen, info = generated_losses(y, roi_map[case_id], tf_grad[case_id], kernel)
        gen.backward()
        gen_norm = torch.nn.utils.clip_grad_norm_(params, 1e9)
        scale = (0.05 * photo_norm / gen_norm.clamp_min(1e-12)).clamp(max=1).detach()
        for param, saved in zip(params, photo_grads):
            if saved is None:
                if param.grad is not None:
                    param.grad.mul_(scale)
                continue
            if param.grad is None:
                param.grad = saved
            else:
                param.grad.mul_(scale).add_(saved)
        grad_norm = torch.nn.utils.clip_grad_norm_(params, 1, error_if_nonfinite=True)
        print("SMOKE_WIDE", json.dumps(dict(shape=meta["shape"], case=case_id, photo_grad_norm=float(photo_norm), gen_grad_norm=float(gen_norm), gen_grad_scale=float(scale), grad_norm=float(grad_norm), peak_gib=torch.cuda.max_memory_allocated() / 2**30, **info)), flush=True)
        atomic_json(RUN / "status.json", dict(state="smoke_passed", peak_gib=torch.cuda.max_memory_allocated() / 2**30))
        writer.close()
        return
    baseline = None
    if start == 0 and not (RUN / "eval_000000" / "metrics.json").exists():
        baseline = evaluate(vae, net, validation, roi_map, writer, 0, official_photo)
    elif (RUN / "eval_000000" / "metrics.json").exists():
        baseline = json.loads((RUN / "eval_000000" / "metrics.json").read_text())
    log_path = RUN / "metrics.jsonl"
    step = start
    recent_mse = []
    with log_path.open("a", buffering=1) as log:
        for step in range(start + 1, steps + 1):
            if STOP:
                break
            t = time.perf_counter()
            gc.collect()
            torch.cuda.reset_peak_memory_stats()
            opt.zero_grad(set_to_none=True)
            lr = LR * min(1.0, step / 50)
            for group in opt.param_groups:
                group["lr"] = lr
            x, meta = photo_sample(step)
            x = x.cuda()
            with torch.no_grad():
                z = vae.encode(x).latent_dist.mode()
            y = decode_photo(vae, z)
            target = x[:, :3, 0]
            err = y[:, :3] - target
            mse = err.square().mean()
            per = perceptual_patches(net, y[:, :3], target)
            coarse = F.avg_pool2d(err, 8).square().mean()
            alpha = (y[:, 3] - x[:, 3, 0]).square().mean()
            photo = mse + 0.05 * per + 0.25 * coarse + 0.1 * alpha
            if not torch.isfinite(photo):
                raise RuntimeError("nonfinite photo loss")
            photo.backward()
            photo_grads = [p.grad.detach().clone() if p.grad is not None else None for p in params]
            photo_norm = torch.nn.utils.clip_grad_norm_(params, 1e9)
            photo_info = dict(mse=float(mse.detach()), lpips=float(per.detach()), coarse=float(coarse.detach()), alpha=float(alpha.detach()), photo=float(photo.detach()), photo_grad_norm=float(photo_norm))
            del x, z, y, target, err, mse, per, coarse, alpha, photo
            opt.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            case_id = TRAIN_IDS[(step - 1) % len(TRAIN_IDS)]
            z = load_latent(vae, case_id, 1024)
            y = decode_generated(vae, z)
            gen, gen_info = generated_losses(y, roi_map[case_id], tf_grad[case_id], kernel)
            if not torch.isfinite(gen):
                raise RuntimeError("nonfinite generated loss")
            gen.backward()
            gen_norm = torch.nn.utils.clip_grad_norm_(params, 1e9)
            # The raw spectral gradient is much larger than the photo gradient. Cap its share at
            # 5% of the photo gradient. A 50% share walked the decoder off the photo manifold by step 21.
            scale = (0.05 * photo_norm / gen_norm.clamp_min(1e-12)).clamp(max=1).detach()
            for param, saved in zip(params, photo_grads):
                if saved is None:
                    if param.grad is not None:
                        param.grad.mul_(scale)
                    continue
                if param.grad is None:
                    param.grad = saved
                else:
                    param.grad.mul_(scale).add_(saved)
            peak_abs = max((float(p.grad.detach().abs().max()) for p in params if p.grad is not None), default=0.0)
            grad_norm = torch.nn.utils.clip_grad_norm_(params, 1, error_if_nonfinite=True)
            gen_info = dict(gen_info, gen_grad_norm=float(gen_norm), gen_grad_scale=float(scale), grad_abs_max=peak_abs)
            opt.step()
            torch.cuda.synchronize()
            row = dict(
                step=step,
                lr=lr,
                case=case_id,
                **meta,
                **photo_info,
                **gen_info,
                gen=float(gen.detach()),
                grad_norm=float(grad_norm),
                wall_seconds=time.perf_counter() - t,
                peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
            )
            log.write(json.dumps(row) + "\n")
            atomic_json(RUN / "status.json", dict(state="training", target=steps, **{k: v for k, v in row.items() if not isinstance(v, (list, dict))}))
            print("STEP", json.dumps(row), flush=True)
            for key, value in row.items():
                if isinstance(value, (int, float)):
                    writer.add_scalar("train/" + key, value, step)
            del z, y, gen
            recent_mse.append(row["mse"])
            if len(recent_mse) > 10:
                recent_mse.pop(0)
            if step >= 20 and len(recent_mse) == 10 and float(np.median(recent_mse)) > 0.02:
                save_checkpoint(vae, opt, step)
                reason = f"median photo mse {float(np.median(recent_mse)):.4f} over steps {step - 9}-{step}"
                atomic_json(RUN / "status.json", dict(state="early_stop", step=step, reason=reason))
                print("EARLY_STOP", reason, flush=True)
                break
            if step % 250 == 0 or step == steps or STOP:
                save_checkpoint(vae, opt, step)
            if step % 250 == 0 or step == steps:
                summary = evaluate(vae, net, validation, roi_map, writer, step, official_photo)
                stop, reason = should_stop(summary, baseline)
                if stop:
                    atomic_json(RUN / "status.json", dict(state="early_stop", step=step, reason=reason))
                    print("EARLY_STOP", reason, flush=True)
                    break
            if STOP:
                save_checkpoint(vae, opt, step)
                break
    frozen_ok = all(not p.requires_grad or name.startswith(TRAINABLE_PREFIXES) for name, p in vae.named_parameters())
    prev = json.loads((RUN / "status.json").read_text())
    state = "stopped" if STOP else prev.get("state", "complete")
    if state == "training":
        state = "complete"
    payload = dict(state=state, step=step, frozen_scope_ok=frozen_ok)
    if prev.get("reason"):
        payload["reason"] = prev["reason"]
    atomic_json(RUN / "status.json", payload)
    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    try:
        main(args.steps, args.smoke)
    except Exception:
        RUN.mkdir(parents=True, exist_ok=True)
        atomic_json(RUN / "failure.json", dict(traceback=traceback.format_exc()))
        atomic_json(RUN / "status.json", dict(state="failed"))
        raise

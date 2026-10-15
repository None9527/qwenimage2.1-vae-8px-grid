"""Re-measure the first SiLU that births the 8px and 4px components.

Loads the October 12 forehead latent and prints the ratio table from the
note: up_blocks.2 axis share before SiLU, after SiLU, and after conv1;
up_blocks.3 diagonal share at the same three points.

    PYTHONPATH=. python experiments/localize/trace_silu.py
    VAE8_EXTRA=/path/to/step750.safetensors PYTHONPATH=. python experiments/localize/trace_silu.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import torch
from diffusers import AutoencoderKLQwenImage21, QwenImage21Pipeline
from safetensors.torch import load_file

from vae8px.localize import phase, stat
from vae8px.paths import PATHS


def load_latent(vae):
    packed = torch.load(PATHS.forehead_latent, weights_only=True, map_location="cuda")
    unpacked = QwenImage21Pipeline._unpack_latents(packed, 1024, 1024, 16).float()
    mean = torch.tensor(vae.config.latents_mean, device="cuda").view(1, 64, 1, 1, 1)
    std = torch.tensor(vae.config.latents_std, device="cuda").view(1, 64, 1, 1, 1)
    return unpacked * std + mean


@torch.no_grad()
def trace(vae, latent) -> dict:
    captured = {}
    counts = {2: 0, 3: 0}
    handles = []
    for stage in (2, 3):
        block = vae.decoder.up_blocks[stage].resnets[0]

        def keep(module, inputs, output, key):
            captured[key] = stat(output)

        handles.append(block.norm1.register_forward_hook(lambda *args, stage=stage: keep(*args, key=f"s{stage}_norm1")))
        handles.append(block.conv1.register_forward_hook(lambda *args, stage=stage: keep(*args, key=f"s{stage}_conv1")))

        def activation(module, inputs, output, stage=stage):
            counts[stage] += 1
            if counts[stage] != 1:
                return
            key = f"s{stage}_act1"
            captured[key] = stat(output)
            full, _ = phase(output)
            even, _ = phase(output - inputs[0] * 0.5)
            quadratic, _ = phase(inputs[0].square() * 0.25)
            sl = slice(0, 2) if stage == 2 else slice(2, 3)
            even_vec = even[sl].flatten()
            quad_vec = quadratic[sl].flatten()
            captured[key]["even_phase_over_total"] = float(
                torch.linalg.vector_norm(even_vec) / torch.linalg.vector_norm(full[sl])
            )
            captured[key]["even_quadratic_phase_cosine"] = float(
                torch.dot(even_vec, quad_vec) / (torch.linalg.vector_norm(even_vec) * torch.linalg.vector_norm(quad_vec))
            )

        handles.append(block.nonlinearity.register_forward_hook(activation))
    vae.decode(latent, return_dict=False)
    for handle in handles:
        handle.remove()
    return captured


def main():
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    labels = [("official", PATHS.official_vae), ("texture-fix", PATHS.texture_fix)]
    extra = os.environ.get("VAE8_EXTRA")
    if extra:
        labels.append(("extra", extra))
    rows = []
    for label, path in labels:
        vae = AutoencoderKLQwenImage21.from_pretrained(PATHS.texture_fix if label == "extra" else path, torch_dtype=torch.float32)
        if label == "extra":
            vae.load_state_dict(load_file(path))
        vae = vae.cuda().eval().requires_grad_(False)
        latent = load_latent(vae)
        captured = trace(vae, latent)
        rows.append(dict(model=label, nodes=captured))
        print(label, json.dumps(captured), flush=True)
        del vae, latent
        torch.cuda.empty_cache()
    out = PATHS.localize_run / "trace_silu_rerun.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2))
    print("WROTE", out, flush=True)


if __name__ == "__main__":
    main()

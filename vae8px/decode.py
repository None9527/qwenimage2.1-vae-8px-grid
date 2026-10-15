"""Checkpointed, tiled image decode for Qwen-Image 2.1.

The native decoder keeps the whole activation graph. A 1024 decode next to
a photo step does not fit. This path runs the prefix once, then the last
two upsample stages in 512-pixel tiles with a 16-pixel halo. On a single
frame it matches `vae.decode` to numerical noise; the follow-up smoke test
requires the max absolute difference to be 0.
"""
from __future__ import annotations

import torch
from torch.utils.checkpoint import checkpoint


def _call(fn, *args):
    if torch.is_grad_enabled():
        return checkpoint(fn, *args, use_reentrant=False, preserve_rng_state=False)
    return fn(*args)


def up_block(block, hidden):
    residual = hidden.clone()
    for resnet in block.resnets:
        hidden = _call(lambda value, resnet=resnet: resnet(value), hidden)
    if block.upsampler is not None:
        hidden = _call(block.upsampler, hidden)
    if block.avg_shortcut is not None:
        hidden = hidden + block.avg_shortcut(residual, first_chunk=True)
    return hidden


def mid_forward(mid, hidden):
    hidden = _call(mid.resnets[0], hidden)
    for attention, resnet in zip(mid.attentions, mid.resnets[1:]):
        if attention is not None:
            hidden = _call(attention, hidden)
        hidden = _call(resnet, hidden)
    return hidden


def decoder_features(vae, latent):
    hidden = vae.post_quant_conv(latent)
    hidden = _call(vae.decoder.conv_in, hidden)
    hidden = _call(lambda value: mid_forward(vae.decoder.mid_block, value), hidden)
    for block in vae.decoder.up_blocks[:3]:
        hidden = _call(lambda value, block=block: up_block(block, value), hidden)
    return hidden


def suffix_forward(decoder, hidden):
    for block in decoder.up_blocks[3:]:
        hidden = block(hidden, first_chunk=True)
    hidden = decoder.nonlinearity(decoder.norm_out(hidden))
    return decoder.conv_out(hidden).clamp(-1, 1)


def tiled_suffix(decoder, features, tile_output=512, halo_input=16, checkpointed=True):
    # Suffix only has local ops and upsamples by exactly 2.
    tile = tile_output // 2
    height, width = features.shape[-2:]
    bands = []
    for top in range(0, height, tile):
        columns = []
        for left in range(0, width, tile):
            bottom = min(top + tile, height)
            right = min(left + tile, width)
            y0 = max(top - halo_input, 0)
            x0 = max(left - halo_input, 0)
            y1 = min(bottom + halo_input, height)
            x1 = min(right + halo_input, width)
            patch = features[:, :, :, y0:y1, x0:x1]
            if checkpointed and torch.is_grad_enabled():
                out = checkpoint(
                    lambda patch: suffix_forward(decoder, patch),
                    patch,
                    use_reentrant=False,
                    preserve_rng_state=False,
                )
            else:
                out = suffix_forward(decoder, patch)
            out = out[:, :, :, (top - y0) * 2 : (bottom - y0) * 2, (left - x0) * 2 : (right - x0) * 2]
            columns.append(out)
        bands.append(torch.cat(columns, -1))
    return torch.cat(bands, -2)


def prefix_assertions(vae) -> None:
    # Image-only first frame. The suffix has no attention and no spatial norm,
    # which is what makes the tiles independent.
    assert vae.config.patch_size is None and vae.config.is_residual
    assert all(type(block).__name__ == "QwenImage21ResidualUpBlock" for block in vae.decoder.up_blocks)
    forbidden = {"QwenImage21AttentionBlock", "GroupNorm", "BatchNorm2d"}
    for block in vae.decoder.up_blocks[3:]:
        for module in block.modules():
            assert type(module).__name__ not in forbidden


def decode_image(vae, latent):
    features = decoder_features(vae, latent)
    if torch.is_grad_enabled():
        assert features.requires_grad
    return tiled_suffix(vae.decoder, features, 512, 16, True)

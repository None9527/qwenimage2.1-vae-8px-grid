# Copyright (c) 2020 Patrick Esser and Robin Rombach and Björn Ommer
#
# PatchGAN from pytorch-CycleGAN-and-pix2pix (CompVis / junyanz).
# MIT license: see THIRD_PARTY.md. use_actnorm is unused; arms C and A
# fuse the BatchNorm into the preceding convolution and then run eval.

import functools

import torch.nn as nn

ActNorm = None


def weights_init(module):
    classname = module.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(module.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(module.weight.data, 1.0, 0.02)
        nn.init.constant_(module.bias.data, 0)


class NLayerDiscriminator(nn.Module):
    """PatchGAN as in pix2pix. Receptive field is 70 at n_layers=3, ndf=64."""

    def __init__(self, input_nc=3, ndf=64, n_layers=3, use_actnorm=False):
        super().__init__()
        norm_layer = nn.BatchNorm2d if not use_actnorm else ActNorm
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d
        kernel = 4
        pad = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kernel, stride=2, padding=pad),
            nn.LeakyReLU(0.2, True),
        ]
        filters = 1
        previous = 1
        for layer in range(1, n_layers):
            previous = filters
            filters = min(2**layer, 8)
            sequence += [
                nn.Conv2d(ndf * previous, ndf * filters, kernel_size=kernel, stride=2, padding=pad, bias=use_bias),
                norm_layer(ndf * filters),
                nn.LeakyReLU(0.2, True),
            ]
        previous = filters
        filters = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * previous, ndf * filters, kernel_size=kernel, stride=1, padding=pad, bias=use_bias),
            norm_layer(ndf * filters),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * filters, 1, kernel_size=kernel, stride=1, padding=pad),
        ]
        self.main = nn.Sequential(*sequence)

    def forward(self, image):
        return self.main(image)

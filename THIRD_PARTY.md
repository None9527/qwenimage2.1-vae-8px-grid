# Third-party code

## PatchGAN

`vae8px/patchgan.py` is the NLayerDiscriminator from
[pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py),
originally from the taming-transformers / CompVis implementation.

Copyright (c) 2020 Patrick Esser and Robin Rombach and Björn Ommer

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Seraena

Arms B and AB import `PatchDiscWithContext` from the separate `seraena`
package. It is not vendored here.

## Qwen-Image and Texture-Fix

Training reads the Qwen-Image 2.1 VAE and
[madebyollin/texture-fix-vae-for-qwen-image-2.1](https://huggingface.co/madebyollin/texture-fix-vae-for-qwen-image-2.1).
Those weights are not in this repository. They stay under the Qwen Research
License, non-commercial research. A released derivative has to carry LICENSE
and NOTICE, say Built with Qwen or Improved using Qwen, and not use Qwen as
its primary name. Texture-Fix's NOTICE attributes the earlier decoder
finetune to madebyollin in 2026.

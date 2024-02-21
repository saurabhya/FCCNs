import os
import time
import sys
import math
import torch
import torchvision
from prettytable import PrettyTable


def rgb_to_hsv_mine(image: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    if not isinstance(image, torch.Tensor):
        raise TypeError(
            "Input type is not a torch.Tensor. Got {}".format(type(image)))

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(
            "Input size must have a shape of (*, 3, H, W). Got {}".format(
                image.shape)
        )

    maxc, _ = image.max(-3)
    maxc_mask = image == maxc.unsqueeze(-3)
    _, max_indices = ((maxc_mask.cumsum(-3) == 1) & maxc_mask).max(-3)
    minc: torch.Tensor = image.min(-3)[0]

    v: torch.Tensor = maxc  # brightness

    deltac: torch.Tensor = maxc - minc
    # s: torch.Tensor = deltac
    s: torch.Tensor = deltac / (maxc + eps)

    # avoid division by zero
    deltac = torch.where(
        deltac == 0,
        torch.ones_like(deltac, device=deltac.device, dtype=deltac.dtype),
        deltac,
    )

    maxc_tmp = maxc.unsqueeze(-3) - image
    rc: torch.Tensor = maxc_tmp[..., 0, :, :]
    gc: torch.Tensor = maxc_tmp[..., 1, :, :]
    bc: torch.Tensor = maxc_tmp[..., 2, :, :]

    h = torch.stack([bc - gc, 2.0 * deltac + rc - bc,
                    4.0 * deltac + gc - rc], dim=-3)

    h = torch.gather(h, dim=-3, index=max_indices[..., None, :, :])
    h = h.squeeze(-3)
    h = h / deltac

    h = (h / 6.0) % 1.0

    h = 2 * math.pi * h

    return torch.stack([h, s, v], dim=-3)


def hsv_to_rgb(image: torch.Tensor) -> torch.Tensor:
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(image)}")

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(
            f"Input size must have a shape of (*, 3, H, W). Got {image.shape}"
        )

    h: torch.Tensor = image[..., 0, :, :] / (2 * math.pi)
    s: torch.Tensor = image[..., 1, :, :]
    v: torch.Tensor = image[..., 2, :, :]

    hi: torch.Tensor = torch.floor(h * 6) % 6
    f: torch.Tensor = ((h * 6) % 6) - hi
    one: torch.Tensor = torch.tensor(
        1.0, device=image.device, dtype=image.dtype)
    p: torch.Tensor = v * (one - s)
    q: torch.Tensor = v * (one - f * s)
    t: torch.Tensor = v * (one - (one - f) * s)

    hi = hi.long()
    indices: torch.Tensor = torch.stack([hi, hi + 6, hi + 12], dim=-3)
    out = torch.stack((v, q, p, p, t, v, t, v, v, q,
                      p, p, p, p, t, v, v, q), dim=-3)
    out = torch.gather(out, -3, indices)

    return out


# Function for pytorch transforms


class ToHSV(object):
    def __call__(self, pic):
        """RGB image to HSV image"""
        return rgb_to_hsv_mine(pic)

    def __repr__(self):
        return self.__class__.__name__ + "()"


class ToRGB(object):
    def __call__(self, img):
        """HSV image to RGB image"""
        return hsv_to_rgb(img)

    def __repr__(self) -> str:
        return self.__class__.__name__ + "()"


class ToiRGB(object):
    def __call__(self, img):
        if not img.is_complex():
            raise ValueError(f"Input should be a complex tensor")

        real, imag = img.real, img.imag

        return (hsv_to_rgb(real)).type(torch.complex64) + 1j * (hsv_to_rgb(imag)).type(
            torch.complex64
        )

    def __repr__(self):
        return self.__class__.__name__ + "()"


# Function for complex conversion


class ToComplex(object):
    def __call__(self, img):
        hue = img[..., 0, :, :]
        sat = img[..., 1, :, :]
        val = img[..., 2, :, :]

        real_1 = sat * hue
        real_2 = sat * torch.cos(hue)
        real_3 = val

        imag_1 = val
        imag_2 = sat * torch.sin(hue)
        imag_3 = sat

        real = torch.stack([real_1, real_2, real_3], dim=-3)
        imag = torch.stack([imag_1, imag_2, imag_3], dim=-3)

        comp_tensor = torch.complex(real, imag)

        assert comp_tensor.dtype == torch.complex64
        return comp_tensor

    def __repr__(self):
        return self.__class__.__name__ + "()"

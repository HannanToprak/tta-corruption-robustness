import random

import numpy as np
from PIL import Image, ImageEnhance, ImageOps
import torch
import torch.nn.functional as F
from torchvision import transforms


def int_parameter(level, maxval):
    return int(level * maxval / 10)


def float_parameter(level, maxval):
    return float(level) * maxval / 10.0


def sample_level(n):
    return np.random.uniform(low=0.1, high=n)


def autocontrast(pil_img, level):
    return ImageOps.autocontrast(pil_img)


def equalize(pil_img, level):
    return ImageOps.equalize(pil_img)


def posterize(pil_img, level):
    level = int_parameter(sample_level(level), 4)
    return ImageOps.posterize(pil_img, 4 - level)


def rotate(pil_img, level):
    degrees = int_parameter(sample_level(level), 30)
    if random.random() > 0.5:
        degrees = -degrees
    return pil_img.rotate(degrees)


def solarize(pil_img, level):
    level = int_parameter(sample_level(level), 256)
    return ImageOps.solarize(pil_img, 256 - level)


def shear_x(pil_img, level):
    level = float_parameter(sample_level(level), 0.3)
    if random.random() > 0.5:
        level = -level
    return pil_img.transform(
        pil_img.size,
        Image.AFFINE,
        (1, level, 0, 0, 1, 0),
        resample=Image.BILINEAR,
    )


def shear_y(pil_img, level):
    level = float_parameter(sample_level(level), 0.3)
    if random.random() > 0.5:
        level = -level
    return pil_img.transform(
        pil_img.size,
        Image.AFFINE,
        (1, 0, 0, level, 1, 0),
        resample=Image.BILINEAR,
    )


def translate_x(pil_img, level):
    level = int_parameter(sample_level(level), pil_img.size[0] / 3)
    if random.random() > 0.5:
        level = -level
    return pil_img.transform(
        pil_img.size,
        Image.AFFINE,
        (1, 0, level, 0, 1, 0),
        resample=Image.BILINEAR,
    )


def translate_y(pil_img, level):
    level = int_parameter(sample_level(level), pil_img.size[1] / 3)
    if random.random() > 0.5:
        level = -level
    return pil_img.transform(
        pil_img.size,
        Image.AFFINE,
        (1, 0, 0, 0, 1, level),
        resample=Image.BILINEAR,
    )


def color(pil_img, level):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Color(pil_img).enhance(level)


def contrast(pil_img, level):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Contrast(pil_img).enhance(level)


def brightness(pil_img, level):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Brightness(pil_img).enhance(level)


def sharpness(pil_img, level):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Sharpness(pil_img).enhance(level)


AUGMENTATIONS = [
    autocontrast,
    equalize,
    posterize,
    rotate,
    solarize,
    shear_x,
    shear_y,
    translate_x,
    translate_y,
    color,
    contrast,
    brightness,
    sharpness,
]


def augmix(
    image,
    preprocess,
    severity=3,
    width=3,
    depth=-1,
    alpha=1.0,
):
    """
    Full AugMix image generation.

    Args:
        image: PIL image
        preprocess: transform converting PIL image to tensor
        severity: augmentation strength
        width: number of augmentation chains
        depth: chain depth; -1 means randomly sample 1-3
        alpha: Dirichlet/Beta mixing parameter

    Returns:
        Mixed augmented tensor
    """

    ws = np.float32(
        np.random.dirichlet([alpha] * width) #The code samples mixing weights using a Dirichlet distribution 
    )
    m = np.float32(
        np.random.beta(alpha, alpha) # samples a mixing coefficient using a Beta distribution
    )

    mix = torch.zeros_like(preprocess(image))

    for i in range(width):
        image_aug = image.copy()

        depth_i = depth
        if depth_i == -1:
            depth_i = np.random.randint(1, 4)

        for _ in range(depth_i):
            op = random.choice(AUGMENTATIONS)
            image_aug = op(image_aug, severity)

        mix += ws[i] * preprocess(image_aug)

    mixed = (1 - m) * preprocess(image) + m * mix

    return mixed


class AugMixDataset(torch.utils.data.Dataset):
    """
    Dataset wrapper for AugMix training.

    For each image, returns:
        clean image
        augmented image 1
        augmented image 2
        label

    This is needed for Jensen-Shannon consistency loss.
    """

    def __init__(
        self,
        dataset,
        preprocess,
        severity=3,
        width=3,
        depth=-1,
        alpha=1.0,
    ):
        self.dataset = dataset
        self.preprocess = preprocess
        self.severity = severity
        self.width = width
        self.depth = depth
        self.alpha = alpha

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, label = self.dataset[index]

        clean = self.preprocess(image)

        aug1 = augmix(
            image=image,
            preprocess=self.preprocess,
            severity=self.severity,
            width=self.width,
            depth=self.depth,
            alpha=self.alpha,
        )

        aug2 = augmix(
            image=image,
            preprocess=self.preprocess,
            severity=self.severity,
            width=self.width,
            depth=self.depth,
            alpha=self.alpha,
        )

        return clean, aug1, aug2, label


def augmix_jsd_loss(
    logits_clean,
    logits_aug1,
    logits_aug2,
    labels,
    criterion,
    jsd_weight=12.0,
):
    """
    Computes AugMix loss.

    Total loss:
        CE(clean, label) + jsd_weight * JSD(p_clean, p_aug1, p_aug2)
    """

    ce_loss = criterion(logits_clean, labels)

    p_clean = F.softmax(logits_clean, dim=1)
    p_aug1 = F.softmax(logits_aug1, dim=1)
    p_aug2 = F.softmax(logits_aug2, dim=1)

    p_mixture = torch.clamp(
        (p_clean + p_aug1 + p_aug2) / 3.0,
        min=1e-7,
        max=1.0,
    )

    jsd = (
        F.kl_div(
            F.log_softmax(logits_clean, dim=1),
            p_mixture,
            reduction="batchmean",
        )
        + F.kl_div(
            F.log_softmax(logits_aug1, dim=1),
            p_mixture,
            reduction="batchmean",
        )
        + F.kl_div(
            F.log_softmax(logits_aug2, dim=1),
            p_mixture,
            reduction="batchmean",
        )
    ) / 3.0

    loss = ce_loss + jsd_weight * jsd

    return loss, ce_loss, jsd

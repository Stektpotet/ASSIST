import torch.nn as nn
import torchvision.transforms as transforms

import torch
import random

import numbers
from torchvision.transforms import functional as F

import numpy as np
from PIL import Image


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class ColorJitterExcludingMask(object):
    """Randomly change the brightness, contrast and saturation of an image.

    Args:
        brightness (float or tuple of float (min, max)): How much to jitter brightness.
            brightness_factor is chosen uniformly from [max(0, 1 - brightness), 1 + brightness]
            or the given [min, max]. Should be non negative numbers.
        contrast (float or tuple of float (min, max)): How much to jitter contrast.
            contrast_factor is chosen uniformly from [max(0, 1 - contrast), 1 + contrast]
            or the given [min, max]. Should be non negative numbers.
        saturation (float or tuple of float (min, max)): How much to jitter saturation.
            saturation_factor is chosen uniformly from [max(0, 1 - saturation), 1 + saturation]
            or the given [min, max]. Should be non negative numbers.
        hue (float or tuple of float (min, max)): How much to jitter hue.
            hue_factor is chosen uniformly from [-hue, hue] or the given [min, max].
            Should have 0<= hue <= 0.5 or -0.5 <= min <= max <= 0.5.
    """
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.brightness = self._check_input(brightness, 'brightness')
        self.contrast = self._check_input(contrast, 'contrast')
        self.saturation = self._check_input(saturation, 'saturation')
        self.hue = self._check_input(hue, 'hue', center=0, bound=(-0.5, 0.5),
                                     clip_first_on_zero=False)

    def _check_input(self, value, name, center=1, bound=(0, float('inf')), clip_first_on_zero=True):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError("If {} is a single number, it must be non negative.".format(name))
            value = [center - value, center + value]
            if clip_first_on_zero:
                value[0] = max(value[0], 0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError("{} values should be between {}".format(name, bound))
        else:
            raise TypeError("{} should be a single number or a list/tuple with lenght 2.".format(name))

        # if value is 0 or (1., 1.) for brightness/contrast/saturation
        # or (0., 0.) for hue, do nothing
        if value[0] == value[1] == center:
            value = None
        return value

    @staticmethod
    def get_params(brightness, contrast, saturation, hue):
        """Get a randomized transform to be applied on image.

        Arguments are same as that of __init__.

        Returns:
            Transform which randomly adjusts brightness, contrast and
            saturation in a random order.
        """
        transforms_list = []

        if brightness is not None:
            brightness_factor = float(torch.empty(1).uniform_(brightness[0], brightness[1]))
            transforms_list.append(transforms.Lambda(lambda img: F.adjust_brightness(img, brightness_factor)))

        if contrast is not None:
            contrast_factor = float(torch.empty(1).uniform_(contrast[0], contrast[1]))
            transforms_list.append(transforms.Lambda(lambda img: F.adjust_contrast(img, contrast_factor)))

        if saturation is not None:
            saturation_factor = float(torch.empty(1).uniform_(saturation[0], saturation[1]))
            transforms_list.append(transforms.Lambda(lambda img: F.adjust_saturation(img, saturation_factor)))

        if hue is not None:
            hue_factor = float(torch.empty(1).uniform_(hue[0], hue[1]))
            transforms_list.append(transforms.Lambda(lambda img: F.adjust_hue(img, hue_factor)))

        random.shuffle(transforms_list)
        transforms_list = transforms.Compose(transforms_list)

        return transforms_list

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Input image.

        Returns:
            PIL Image: Color jittered image.
        """

        transform = self.get_params(self.brightness, self.contrast,
                                    self.saturation, self.hue)
        img_np = np.asarray(img)
        mask_np = np.where(img_np == 0, 0, 1).astype(np.uint8)
        img = transform(img)

        img_trans_np = np.asarray(img)
        img_np = np.where(mask_np == 1, img_trans_np, img_np)
        img = Image.fromarray(img_np.astype('uint8'), 'RGB')

        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'brightness={0}'.format(self.brightness)
        format_string += ', contrast={0}'.format(self.contrast)
        format_string += ', saturation={0}'.format(self.saturation)
        format_string += ', hue={0})'.format(self.hue)
        return format_string


class NormalizeExcludingMask(object):
    """Normalize a tensor image with mean and standard deviation using only non zero pixels.
    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
    """

    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized Tensor image.
        """
        return self.normalize_excluding_mask(tensor, self.mean, self.std)

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

    def normalize_excluding_mask(self, tensor, mean, std):
        mean = torch.tensor(mean)[:, None, None]
        std = torch.tensor(std)[:, None, None]

        tensor_mask = torch.where(tensor == 0, torch.tensor(0), torch.tensor(1))
        tensor_norm = (tensor - mean) / std
        tensor_norm = tensor_norm * tensor_mask

        return tensor_norm

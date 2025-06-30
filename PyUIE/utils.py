import io

import numpy as np
import torch
from PIL import Image
from collections.abc import Iterable

from path import Path


class DataSetUtils:
    @staticmethod
    def shuffle_array(array, seed=None):
        if seed is None:
            rng = np.random.default_rng()
        else:

            rng = np.random.RandomState(seed)
        array_copy = array.copy()
        rng.shuffle(array_copy)

        return array_copy

    @staticmethod
    def split_array_by_ratios(array, ratios):

        sizes = [int(len(array) * ratio) for ratio in ratios]


        subarrays = np.array_split(array, np.cumsum(sizes)[:-1])

        return subarrays


class ImageUtils:
    @staticmethod
    def chw_tensor_to_rgb_numpy(t):
        if not isinstance(t, torch.Tensor) or len(t.shape) != 3:
            raise ValueError("Input must be a chw tensor")
        return (t * 255).clamp(0, 255).byte().permute(1, 2, 0).numpy()

    @staticmethod
    def chw_tensor_to_rgb_image(t):
        return Image.fromarray(ImageUtils.chw_tensor_to_rgb_numpy(t))

    @staticmethod
    def save_image(image_tensor, path):
        Path(path).parent.makedirs_p()
        ImageUtils.chw_tensor_to_rgb_image(image_tensor).save(path)

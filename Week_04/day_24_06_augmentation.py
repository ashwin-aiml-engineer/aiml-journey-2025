"""Day 24.06 — Simple image augmentations (flip, rotate, color jitter)
Run time: ~12 minutes

- Uses Pillow + numpy to apply a handful of common augmentations
"""

from PIL import Image, ImageEnhance
import numpy as np


def flip_lr(arr):
    return np.fliplr(arr)


def rotate(arr, angle):
    img = Image.fromarray(arr)
    return np.array(img.rotate(angle))


def color_jitter(arr, brightness=1.0, contrast=1.0):
    img = Image.fromarray(arr)
    img = ImageEnhance.Brightness(img).enhance(brightness)
    img = ImageEnhance.Contrast(img).enhance(contrast)
    return np.array(img)

if __name__ == '__main__':
    img = Image.new('RGB', (64, 64), color=(180, 120, 60))
    arr = np.array(img)
    print('orig shape', arr.shape)
    print('flip shape', flip_lr(arr).shape)
    print('rotate shape', rotate(arr, 30).shape)
    print('jitter shape', color_jitter(arr, 1.2, 0.9).shape)

    # Exercises:
    # - Chain augmentations randomly to create a small augmentation pipeline.
    # - Add random crop and color channel permutation.
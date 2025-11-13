"""Day 24.01 — Image basics: representation & color spaces
Run time: ~10 minutes

- Show RGB, grayscale, HSV conversions using Pillow and numpy
- Lightweight and safe (no heavy deps)
"""

from PIL import Image
import numpy as np


def to_grayscale(arr):
    # arr: HxWx3 RGB in [0,255]
    return (0.2989 * arr[..., 0] + 0.5870 * arr[..., 1] + 0.1140 * arr[..., 2]).astype(np.uint8)


def rgb_to_hsv(arr):
    # use PIL convert for reliability
    img = Image.fromarray(arr)
    return np.array(img.convert('HSV'))

if __name__ == '__main__':
    # Create a small synthetic RGB image
    img = Image.new('RGB', (64, 48), color=(10, 120, 200))
    arr = np.array(img)
    print('RGB shape:', arr.shape, 'dtype:', arr.dtype)

    gray = to_grayscale(arr)
    print('Grayscale shape:', gray.shape, 'dtype:', gray.dtype)

    hsv = rgb_to_hsv(arr)
    print('HSV shape:', hsv.shape)

    # Exercises:
    # - Try normalizing arr to [0,1] and standardizing (mean=0,std=1) per channel.
    # - Load a JPEG file and compare histogram of R,G,B channels.
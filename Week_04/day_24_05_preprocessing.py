"""Day 24.05 — Image loading, resize, normalization, batching
Run time: ~12 minutes

- Safe image loader using Pillow, resize, normalize to [0,1], and create batches
"""

from PIL import Image
import numpy as np


def load_and_preprocess(path, size=(96,96)):
    img = Image.open(path).convert('RGB')
    img = img.resize(size)
    arr = np.array(img).astype(np.float32) / 255.0
    # per-channel mean/std normalization (example values)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    arr = (arr - mean) / std
    return arr


def make_batch(list_of_paths, size=(96,96)):
    batch = [load_and_preprocess(p, size) for p in list_of_paths]
    return np.stack(batch, axis=0)

if __name__ == '__main__':
    # demo with synthetic image saved to disk
    demo = Image.new('RGB', (128, 128), color=(120, 60, 200))
    demo.save('demo_img.jpg')
    b = make_batch(['demo_img.jpg'] * 4)
    print('Batch shape:', b.shape, 'dtype:', b.dtype)

    # Exercises:
    # - Replace normalization with channel-wise min/max scaling.
    # - Implement a generator that yields batches from a folder of images.
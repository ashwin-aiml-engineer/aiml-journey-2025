"""Day 26.03 — Dataset preparation: formatting, splits, generator
Run time: ~12 minutes

- Small utilities to create train/val/test splits and a simple generator
"""

import random
import json
from pathlib import Path


def write_splits(file_list, out_dir='data_splits', ratios=(0.8, 0.1, 0.1), seed=42):
    random.seed(seed)
    files = list(file_list)
    random.shuffle(files)
    n = len(files)
    t = int(ratios[0] * n)
    v = int((ratios[0] + ratios[1]) * n)
    splits = {'train': files[:t], 'val': files[t:v], 'test': files[v:]}
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    for k, vfiles in splits.items():
        with open(Path(out_dir) / f'{k}.json', 'w', encoding='utf-8') as f:
            json.dump(vfiles, f)
    print('Wrote splits to', out_dir)


def simple_generator(list_of_paths, batch_size=8):
    # yields batches of file paths (user should load/transform inside loop)
    for i in range(0, len(list_of_paths), batch_size):
        yield list_of_paths[i:i+batch_size]

if __name__ == '__main__':
    files = [f'image_{i}.jpg' for i in range(100)]
    write_splits(files)
    gen = simple_generator(files, batch_size=16)
    print('First batch example:', next(gen))

    # Exercises:
    # - Extend write_splits to stratify by class labels.
    # - Implement an on-disk generator that yields preprocessed tensors.
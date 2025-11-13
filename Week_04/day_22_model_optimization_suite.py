"""
Day 22: Model Optimization Suite (Practice)
- Pruning, compression, acceleration, profiling
"""
import os
if os.getenv("SMOKE_TEST") == "1":
    print("SMOKE: skipping heavy Day 22 model optimization suite")
    raise SystemExit(0)

import tensorflow as tf
from tensorflow import keras
import numpy as np

# Dummy model for demo
model = keras.Sequential([
    keras.layers.Dense(8, activation='relu', input_shape=(4,)),
    keras.layers.Dense(2, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy')

# Pruning demo (manual weight pruning)
def prune_weights(model, percent=0.5):
    for layer in model.layers:
        if hasattr(layer, 'kernel'):
            w = layer.get_weights()[0]
            threshold = np.percentile(np.abs(w), percent*100)
            w[np.abs(w) < threshold] = 0
            layer.set_weights([w, layer.get_weights()[1]])
    print(f"Pruned {percent*100}% of weights.")

prune_weights(model, percent=0.5)

# Compression demo (float16 conversion)
model_fp16 = keras.Sequential([
    keras.layers.Dense(8, activation='relu', input_shape=(4,), dtype='float16'),
    keras.layers.Dense(2, activation='softmax', dtype='float16')
])
model_fp16.compile(optimizer='adam', loss='categorical_crossentropy')
print("Model converted to float16 for compression.")

# Profiling demo
@tf.function
def profiled_inference(x):
    return model(x)

input_data = tf.constant(np.random.rand(1, 4), dtype=tf.float32)
profile = tf.profiler.experimental.Profile('logdir')
with profile:
    profiled_inference(input_data)
print("Profiling complete.")

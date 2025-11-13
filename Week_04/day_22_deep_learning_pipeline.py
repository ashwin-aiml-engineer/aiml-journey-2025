"""
Day 22: Deep Learning Pipeline (TensorFlow/Keras Practice)
- Build, train, optimize, evaluate DNN
"""
import os
if os.getenv("SMOKE_TEST") == "1":
    print("SMOKE: skipping heavy Day 22 deep learning pipeline")
    raise SystemExit(0)

from tensorflow import keras  # type: ignore
from tensorflow.keras import layers  # type: ignore
import numpy as np

# Data
X = np.random.rand(100, 4)
y = keras.utils.to_categorical(np.random.randint(0, 2, 100), 2)

# Model
model = keras.Sequential([
    layers.Dense(8, activation='relu', input_shape=(4,)),
    layers.Dropout(0.2),
    layers.BatchNormalization(),
    layers.Dense(2, activation='softmax')
])
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy'],
)

# Training
history = model.fit(
    X,
    y,
    epochs=10,
    batch_size=16,
    validation_split=0.2,
    verbose=0,
)

# Evaluation
loss, acc = model.evaluate(X, y, verbose=0)
print(f"DNN accuracy: {acc:.2f}, loss: {loss:.2f}")

# Early stopping callback demo
callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)
model.fit(
    X,
    y,
    epochs=10,
    batch_size=16,
    validation_split=0.2,
    callbacks=[callback],
    verbose=0,
)

# Save model
model.save('dnn_model.h5')
print("Model saved as dnn_model.h5")

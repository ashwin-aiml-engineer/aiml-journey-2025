# Day 24 — 12 Quick Exercises (15-min each)

1. Load an image and convert between RGB, HSV, and Grayscale. Plot histograms for each channel.
2. Implement a simple 3x3 Sobel filter using `day_24_02_conv_demo.py` and visualize edges.
3. Compare stride and padding effects by running `day_24_02_conv_demo.py` with different params.
4. Use `day_24_05_preprocessing.py` to build a small batch loader for a folder of images.
5. Chain three augmentations (flip, rotate, brightness jitter) and show 5 samples.
6. Use `day_24_04_transfer_learning.py` to load a pre-trained backbone (if TF installed) and print layer counts.
7. Replace the `dummy_predict` in webcam demo with a simple threshold classifier from a batch of sample images.
8. Write a 10-line plan for PTQ for a MobileNet model (calibration dataset, metrics to collect).
9. Convert a saved small Keras model to TFLite (if TF available) using `day_24_08_tflite_onnx_stubs.py` as guide.
10. Create a minimal local API outline (Flask/FastAPI) that accepts image upload and returns prediction JSON.
11. Measure CPU inference time for a small model (use a synthetic input and time.perf_counter).
12. Sketch deployment differences between Raspberry Pi and Jetson Nano (memory, acceleration, drivers).

Use these exercises as 15-min practical blocks to catch up quickly.
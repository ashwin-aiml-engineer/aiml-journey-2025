# Day 23.08 — Hardware-specific optimization (concise tips)

Run time: ~10 minutes reading + quick checks

CPU
- Use vectorized ops (NumPy) and BLAS-backed libraries.
- Avoid Python loops in hot paths.
- Use int8/fp16 where supported by inference runtimes.

GPU
- Keep large batch sizes to maximize utilization (within memory limits).
- Fuse ops when possible; use framework-provided fused kernels.
- Use mixed precision (FP16) for faster compute on modern GPUs.

Edge / Mobile
- Use TensorFlow Lite or ONNX Runtime Mobile.
- Convert/quantize models to INT8 for smaller size and faster inference.
- Mind memory bandwidth: reduce activation sizes (recompute, checkpointing).

Formats & Tools
- TensorFlow Lite (.tflite), ONNX (.onnx), TorchScript (.pt)
- Use vendor tools for accelerators (e.g., TensorRT, NNAPI, CoreML)

Quick checks
- Profile a model with framework profiler to find bottlenecks.
- Measure inference time with realistic batch sizes and cold starts.

Exercises:
- Try converting a small Keras model to TFLite (if you have TF installed).
- Run ONNX conversion and inspect model graph for removable ops.
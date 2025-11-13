# Day 25.06 — Inference Optimization (concise notes)

Key techniques
- Quantization: PTQ and QAT for Transformers (INT8, FP16)
- ONNX export: convert and run with ONNX Runtime for speed
- Batching: increase throughput, be mindful of latency
- Model sharding & offloading for very large models
- Use token streaming for generation to reduce latency

Quick checklist
- Measure baseline latency on CPU/GPU
- Try FP16/mixed-precision if hardware supports it
- If memory-bound, prefer per-channel quantization for weights

Note: Many experiments require framework support (transformers, onnxruntime, torch).
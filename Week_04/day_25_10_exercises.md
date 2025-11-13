# Day 25 — 12 Quick Exercises (15-min each)

1. Run a `transformers` pipeline (sentiment-analysis) and inspect outputs for two samples.
2. Use `day_25_02_transformer_fundamentals.py` to visualize attention weights for a toy sequence.
3. Tokenization: compare BPE vs whitespace tokenizer on a few sentences.
4. Load a small pre-trained model card on HF Hub and read its license and pipeline tag.
5. Use `day_25_04_pipelines_demo.py` to time a pipeline on a batch of 8 texts (if TF/PyTorch available).
6. Try converting a small model to ONNX (follow `day_25_06_inference_optimization.md`).
7. Build a minimal FastAPI Flask skeleton that serves a text classification pipeline.
8. Measure CPU memory and latency for a pipeline on a synthetic input.
9. Implement a tiny multi-head attention and observe shapes when using 2 heads.
10. Explore `huggingface_hub` API to list tags and downloads for a model.
11. Write a short plan (20 lines) for PTQ of a transformer for CPU inference.
12. Create a fallback heuristic (regex or small classifier) to handle extremely long inputs before tokenization.

Use these exercises as 15-minute practical blocks to get up to speed quickly.
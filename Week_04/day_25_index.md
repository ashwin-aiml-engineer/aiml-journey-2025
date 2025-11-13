# Day 25: Hugging Face Ecosystem — Pre-trained Models

Short practice files (each ~15 minutes) covering the Hugging Face Hub, Transformers fundamentals, tokenization, pipelines, hub integration, optimization and local deployment patterns.

Files:
- `day_25_01_hf_overview.md` — Hub, Transformers, Datasets, Tokenizers, Accelerate, Gradio
- `day_25_02_transformer_fundamentals.py` — attention/self-attention toy demo (numpy)
- `day_25_03_pretrained_models.md` — model categories and when to pick them
- `day_25_04_pipelines_demo.py` — quick pipeline demo using `transformers` if installed (safe stub otherwise)
- `day_25_05_tokenization.py` — tokenizer demo with fallback if `transformers` not installed
- `day_25_06_inference_optimization.md` — tips for quantization/ONNX/Batching
- `day_25_07_hub_integration.py` — download model metadata and show usage patterns (stub if HF libs missing)
- `day_25_08_practical_patterns.md` — API patterns, multi-task inference, error handling
- `day_25_09_local_deploy.md` — local deployment considerations and quick Flask example outline
- `day_25_10_exercises.md` — 12 concise 15-min exercises

How to use
1. From repo root run: `python Week_04\\day_25_02_transformer_fundamentals.py` etc.
2. Many scripts use pure numpy or include safe fallbacks if `transformers` / HF libraries are not installed.
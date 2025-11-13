# Day 25.01 — Hugging Face Ecosystem Overview (concise)

Key components
- Hub: model/dataset/space repository and versioning
- Transformers: model loading, AutoModel/AutoTokenizer, pipeline API
- Datasets: dataset loading, caching, streaming
- Tokenizers: fast tokenizers (Rust-backed), BPE/WordPiece/SentencePiece
- Accelerate: distributed training helper
- Gradio/Spaces: quick demos and web UIs

Quick notes
- Use `pipeline(task, model=...)` for one-line inference experiments.
- Check model card for size, license, and intended task.
- Use Datasets to create efficient preprocessing and streaming pipelines.

Suggested short run
- If `transformers` installed, try a pipeline example (see `day_25_04_pipelines_demo.py`).
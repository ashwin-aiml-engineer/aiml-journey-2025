# Day 25.09 — Local Deployment Considerations (concise)

Local deployment checklist
- Model size and storage: consider compressed/quantized artifacts
- Cold starts: keep a warm process or preload model on startup
- Resource estimates: RAM and CPU for typical model sizes
- Model warming: run a few dummy inferences after load to JIT/allocate caches

Flask/FastAPI sketch
- Accept POST /predict with JSON {"text": "..."}
- Load tokenizer and model once at startup
- Use a threadpool for CPU-bound inference or async for IO

Quick exercise
- Write a 10-line Flask skeleton that loads a tokenizer and returns pipeline output.
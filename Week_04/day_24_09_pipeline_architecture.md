# Day 24.09 — CV Pipeline Architecture (concise checklist)

End-to-end steps
- Image acquisition (camera / upload)
- Validation & sanity checks (shape, corrupted files)
- Preprocessing & augmentation (resize, normalize, augment)
- Model inference (batching, device selection)
- Post-processing (NMS for detection, thresholding for classification)
- Result formatting & visualization
- Caching & storage (avoid reprocessing when possible)
- Monitoring & logging (latency, error rates, input stats)

Operational notes
- Separate preprocessing from model code so you can optimize or replace it.
- Use small, deterministic transforms in production. Keep heavy augmentation for training only.
- For real-time: use async I/O and a lightweight model or hardware acceleration.

Quick exercise
- Draw a minimal diagram (box list) for a webcam-based object detector pipeline.

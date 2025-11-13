Packaging and model flavors — quick guide (10–15 minutes)

Key points
- Serialization formats: joblib/pickle (sklearn), torch.save, tf.keras.save, ONNX for cross-framework
- Model flavor: MLflow flavors let you store models in a framework-agnostic way
- Model signature: define input/output schema to ensure compatibility
- Container packaging: include runtime (Dockerfile) and pinned deps

Mini task
- List the steps to create a Docker image that serves a pickled sklearn model with Flask.

Notes
- Avoid pickle for untrusted inputs; prefer standardized formats (ONNX) where possible.
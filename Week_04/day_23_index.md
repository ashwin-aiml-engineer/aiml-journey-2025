# Day 23: Neural Networks + Quantization Techniques

This folder contains concise (<=15 min) practice files covering advanced neural networks, regularization, training techniques, quantization, compression, and deployment-ready optimization.

Files:
- `day_23_01_ffn_skip.py` — FFN deep dive, dense layers, skip/residual example
- `day_23_02_multitask_ensemble.py` — multi-task and simple ensemble patterns
- `day_23_03_regularization.py` — dropout variants, batch/layer norm, weight decay demo
- `day_23_04_training_techniques.py` — LR schedules, warmup, cyclical LR snippets
- `day_23_05_quant_fundamentals.py` — quantization concepts and FP32->INT8 simulation
- `day_23_06_quant_techniques.py` — weight/activation quantization helpers + calibration
- `day_23_07_model_compression.py` — SVD low-rank demo, pruning sketch
- `day_23_08_hardware_optimization.md` — practical tips for CPU/GPU/edge
- `day_23_09_quant_tradeoffs.py` — quick size vs accuracy experiment scaffold
- `day_23_10_deployment_pipeline.py` — minimal export/convert pipeline notes and stubs
- `day_23_11_exercises.md` — 12 practice exercises (15-min each)

How to use
1. Pick a file matching the topic you want to practice.
2. Open it and read the short explanation at the top.
3. Run it with: `python Week_04/<filename>` (most files only need numpy/scikit-learn).

If you want I can also create lightweight unit tests or Jupyter notebooks for any of these exercises.
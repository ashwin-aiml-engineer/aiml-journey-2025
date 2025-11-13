# Day 23 — 12 Quick Exercises (15-min each)

1. FFN vs Wide Network: Build two small numpy FFNs, one deeper (6 layers x 32) and one wider (2 layers x 256). Compare parameter counts and forward runtime on a small batch.
2. Skip Connection Practice: Modify `day_23_01_ffn_skip.py` to add projection skips when dims differ.
3. Auxiliary Loss: Add an auxiliary head on an intermediate layer and combine losses with a weighted sum.
4. Dropout Variants: Implement DropConnect (multiply weights by dropout mask during training) and compare behavior.
5. BatchNorm vs LayerNorm: Implement LayerNorm and run a tiny stability test with different batch sizes.
6. LR Schedules: Plot step decay, cosine annealing, and cyclical LR for 100 epochs (use matplotlib).
7. PTQ vs QAT thought experiment: Write a short plan (20 lines) that lists steps for PTQ and QAT for a mobile model.
8. Quant Calibration: Run `day_23_05_quant_fundamentals.py` and compute MSE between original and quantized tensors for 8,6,4 bits.
9. Low-rank Compression: Use `day_23_07_model_compression.py` to find the minimum rank achieving <1e-2 MSE.
10. Pruning + Retrain Sketch: Write a 15-line pseudocode to iteratively prune by magnitude and retrain for 3 rounds.
11. Convert to TFLite (if TF installed): Take a small Keras model, save it, convert to TFLite, and run the interpreter on sample input.
12. Edge Benchmark: Measure cold-start latency and steady-state latency for an exported model (use time.perf_counter).

Use the files in this folder as starters; each exercise is intended to be completed in ~15 minutes.
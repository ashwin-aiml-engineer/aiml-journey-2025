# Day 26 — 12 Quick Exercises (15-min each)

1. Freeze backbone and train only a new head for a classification task (feature extraction).
2. Implement gradual unfreezing: unfreeze one block every N epochs and observe val loss.
3. Create train/val/test splits with `day_26_03_data_prep.py` and stratify by class.
4. Using `day_26_04_text_classification.py`, swap head and run a quick 1-epoch training (if transformers installed).
5. Prepare BIO-tagged samples and convert character spans to token spans with a tokenizer.
6. Build one SQuAD-style QA example and compute EM/F1 on sample predictions.
7. Implement a tiny discriminative LR scheme and show different LR values per layer group.
8. Write a 10-line plan for applying LoRA to a transformer and choose target matrices.
9. Run a warmup + cosine schedule and plot LR over 100 steps.
10. Simulate gradient accumulation to achieve effective large batch size and measure steps.
11. Fine-tune a small ViT head on 50 images (demo pipeline sketch).
12. Document a checkpointing strategy and best-model selection criteria for reproducibility.

Use these exercises as 15-minute blocks to get hands-on with fine-tuning concepts.
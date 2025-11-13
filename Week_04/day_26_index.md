# Day 26: Model Fine-tuning — Custom Adaptations

Short practice files (each ~15 minutes) focusing on fine-tuning techniques, data prep, task-specific stubs, PEFT methods, training config, and deployment notes.

Files:
- `day_26_01_finetune_overview.md` — fine-tuning fundamentals and best practices
- `day_26_02_strategies.py` — freezing/unfreezing, discriminative LR pseudocode
- `day_26_03_data_prep.py` — dataset formatting, splits, simple generator
- `day_26_04_text_classification.py` — BERT fine-tune stub or sklearn fallback demo
- `day_26_05_ner_stub.py` — token classification stub and BIO handling notes
- `day_26_06_qa_stub.py` — QA data triplet handling & evaluation sketch
- `day_26_07_cv_finetune.md` — vision fine-tuning notes (ViT, augmentations)
- `day_26_08_peft_lora.md` — PEFT patterns: LoRA, adapters, BitFit quick notes
- `day_26_09_training_config.py` — LR schedule, warmup, grad accumulation sketch
- `day_26_10_exercises.md` — 12 concise 15-min exercises

How to use
1. From repo root run examples with: `python Week_04\\day_26_02_strategies.py` etc.
2. Many scripts include fallbacks so you can run them without heavy deep-learning libs.

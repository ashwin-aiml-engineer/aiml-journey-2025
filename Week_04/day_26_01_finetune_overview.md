# Day 26.01 — Fine-tuning Fundamentals (concise)

Key ideas
- Transfer learning vs fine-tuning: feature extraction freezes backbone; fine-tuning updates weights.
- Freezing/unfreezing: freeze early layers, fine-tune top layers or unfreeze gradually.
- Learning rates: use smaller LR for pre-trained weights, larger for new head.
- PEFT (LoRA/Adapters/BitFit): change few params to adapt efficiently.

Best practices
- Start with feature-extraction (freeze backbone) and train head first.
- Use warmup + small LR for fine-tuning, monitor validation loss closely.
- Keep good checkpoints and use early stopping.

Quick checklist
- Prepare clean dataset, balanced and tokenized/image-processed.
- Choose whether to freeze, gradually unfreeze, or apply PEFT.
- Save a model card documenting dataset and fine-tune settings.
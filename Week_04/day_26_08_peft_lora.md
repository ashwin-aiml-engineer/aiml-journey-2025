# Day 26.08 — Parameter-Efficient Fine-Tuning (PEFT) quick notes

Methods
- Adapters: small bottleneck modules inserted between layers, train only adapters.
- LoRA: inject low-rank updates into weight matrices; keep base weights frozen.
- BitFit: train only bias terms (very cheap but sometimes effective).
- Prefix tuning / prompt tuning: learn task-specific prompts instead of weights.

When to use
- Low compute budget or limited memory (edge devices, fast iteration).
- When dataset is small and full fine-tuning risks overfitting.

Exercise
- Write a 10-line plan for adapting LoRA to a transformer backbone (which matrices to target, rank to try, and LR choices).
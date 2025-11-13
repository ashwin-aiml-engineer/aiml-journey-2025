# Day 26.07 — Computer Vision Fine-tuning (concise)

Key points
- For image classification: replace final head, freeze backbone, train head, then unfreeze last block.
- Use data augmentation and appropriate normalization for pretrained weights.
- For ViT: patch size and resolution matter; consider resizing inputs consistently.

Practical
- Use small learning rate for backbone (e.g., 1e-5) and larger LR for head (1e-3).
- For object detection/segmentation, use task-specific heads and consider anchor/box tuning.

Exercises
- Replace final Dense head of a small model and train on 50 samples to verify pipeline.
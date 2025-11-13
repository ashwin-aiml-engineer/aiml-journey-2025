# Day 25.03 — Pre-trained Model Categories (cheat-sheet)

Language models
- BERT family: encoder-based, excellent for classification/NER
- GPT family: decoder-only, good for generation/completion
- RoBERTa, DistilBERT: variants with training tweaks and smaller sizes

Vision & multimodal
- ViT: Vision Transformer for image classification
- CLIP: image-text embeddings for retrieval and zero-shot
- BLIP/other: image captioning and VQA

Audio
- Wav2Vec, Whisper for speech recognition and tasks

Choosing a model
- Task matters (classification vs generation)
- Size vs latency trade-offs (base vs large)
- License and community benchmarks

Quick note
- Use model cards on HF Hub to confirm input formats and intended tasks.
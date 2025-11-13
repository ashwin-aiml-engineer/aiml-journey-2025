# Day 24: Computer Vision Basics + Local Deployment

Short practice files (each ~15 minutes) covering CV fundamentals, CNNs, transfer learning, preprocessing, augmentation, local deployment and pipeline design.

Files:
- `day_24_01_image_basics.py` — image representation, color spaces, simple conversions
- `day_24_02_conv_demo.py` — small convolution and pooling demo (numpy)
- `day_24_03_cnn_architectures.md` — quick notes on classic CNNs (LeNet, AlexNet, VGG, ResNet, MobileNet)
- `day_24_04_transfer_learning.py` — feature-extraction vs fine-tuning (lightweight stub)
- `day_24_05_preprocessing.py` — image loading, resize, normalization, batching
- `day_24_06_augmentation.py` — basic augmentations (flip, rotate, color jitter)
- `day_24_07_local_deploy_webcam.py` — OpenCV webcam capture + dummy model inference stub
- `day_24_08_tflite_onnx_stubs.py` — conversion stubs for TFLite/ONNX with notes
- `day_24_09_pipeline_architecture.md` — end-to-end CV pipeline design checklist
- `day_24_10_exercises.md` — 12 concise practice exercises

How to use
1. From repo root run: `python Week_04\\day_24_01_image_basics.py` etc.
2. Most scripts use only numpy and Pillow; OpenCV is optional for webcam demo.

If you want, I can run the smoke tests for these files now and create a `day_24_progress.md` tracker.
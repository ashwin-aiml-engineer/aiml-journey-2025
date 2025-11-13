# Day 24.03 — Classic CNN Architectures (cheat-sheet)

LeNet
- Simple early CNN for digit recognition: conv -> pool -> conv -> pool -> fc

AlexNet
- Larger filters, ReLU, dropout, data augmentation; popularized deep CNNs

VGG
- Deep stacks of 3x3 convs with simple architecture rules; heavy parameter count

ResNet
- Residual (skip) connections that ease optimization for very deep nets

Inception
- Mixed-width branches (1x1,3x3,5x5) to capture multi-scale features

MobileNet
- Depthwise separable convolutions for efficiency on mobile/edge

EfficientNet
- Compound scaling (width, depth, resolution) for efficient accuracy trade-offs

Quick exercise
- Compare parameter count intuition: why VGG is large but MobileNet is small.

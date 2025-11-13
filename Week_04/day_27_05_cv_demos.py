"""Day 27.05 — CV demo patterns: image upload -> predict
Run time: ~15 minutes

- Uses Gradio image upload component; dummy classifier if no model available
"""

import os
if os.getenv("SMOKE_TEST") == "1":
    print("SMOKE: skipping Gradio CV demo")
    raise SystemExit(0)

try:
    import gradio as gr
    has_gradio = True
except Exception:
    has_gradio = False

from PIL import Image
import numpy as np


def dummy_image_classifier(img):
    # img: PIL Image or numpy array
    arr = np.array(img.resize((32,32))).mean()
    return 'Bright' if arr > 127 else 'Dark'

if __name__ == '__main__':
    if not has_gradio:
        print('Gradio not installed. Install with pip install gradio to run this demo.')
    else:
        iface = gr.Interface(fn=dummy_image_classifier, inputs=gr.Image(type='pil'), outputs=gr.Label())
        iface.launch(share=False)

    # Exercises:
    # - Replace dummy_image_classifier with a real model (onnx/tflite) for per-image inference.
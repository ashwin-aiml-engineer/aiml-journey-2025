"""Day 24.08 — Conversion stubs and notes for TFLite / ONNX
Run time: ~10 minutes

- Stubs that describe conversion steps. Run only to see instructions if TF/ONNX not installed.
"""

try:
    import os
    if os.getenv("SMOKE_TEST") == "1":
        print("SMOKE: skipping heavy Day 24 TFLite/ONNX stub")
        raise SystemExit(0)

    import tensorflow as tf
    has_tf = True
except Exception:
    has_tf = False

try:
    import onnx
    has_onnx = True
except Exception:
    has_onnx = False

if __name__ == '__main__':
    if has_tf:
        print('TFLite conversion (example):')
        print('converter = tf.lite.TFLiteConverter.from_keras_model(model)')
        print('tflite_model = converter.convert()')
    else:
        print('TensorFlow not available: for TFLite conversion, install tensorflow.')

    if has_onnx:
        print('ONNX export (example):')
        print('tf2onnx.convert.from_keras(model)  # or torch.onnx.export for PyTorch')
    else:
        print('ONNX not available: install onnx and tf2onnx for conversions.')

    print('\nNotes:')
    print('- For TFLite: test both post-training quantization and full integer quantization.')
    print('- For ONNX: verify ops compatibility with target runtime (ONNX Runtime).')

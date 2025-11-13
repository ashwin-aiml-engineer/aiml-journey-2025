"""Day 23.10 — Deployment-ready optimization pipeline (concise)
Run time: ~12 minutes

- Shows the minimal steps for export -> convert -> optimize
- Includes stubs for TFLite conversion and basic advice
"""

import os


def export_keras_model_stub(model, out_path='model.keras'):
    # in real use: model.save(out_path)
    open(out_path, 'w').write('dummy model')
    return out_path


def convert_to_tflite_stub(keras_path, tflite_path='model.tflite'):
    # in real use: use tensorflow.lite.TFLiteConverter
    open(tflite_path, 'w').write('dummy tflite')
    return tflite_path

if __name__ == '__main__':
    kp = export_keras_model_stub(None, 'day23_model.keras')
    tp = convert_to_tflite_stub(kp, 'day23_model.tflite')
    print('Exported and converted (stubs):', kp, '->', tp)

    # Exercises:
    # - Replace stubs with real tf.keras model.save and TFLiteConverter (if TF installed).
    # - Measure tflite file size and run inference with tflite-interpreter.
"""Day 24.04 — Transfer learning: feature extraction vs fine-tuning (concise)
Run time: ~15 minutes

- Shows pattern for using a pre-trained model as a feature extractor or fine-tuning last layers.
- Uses TF/Keras if available; otherwise demonstrates the control flow with stubs.
"""

try:
    import os
    if os.getenv("SMOKE_TEST") == "1":
        print("SMOKE: skipping heavy Day 24 transfer learning demo")
        raise SystemExit(0)

    from tensorflow import keras
    has_tf = True
except Exception:
    has_tf = False

if has_tf:
    def run_demo():
        base = keras.applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=(96,96,3))
        base.trainable = False  # feature extraction
        x = keras.layers.GlobalAveragePooling2D()(base.output)
        out = keras.layers.Dense(10, activation='softmax')(x)
        model = keras.Model(base.input, out)
        model.summary()
        print('Feature-extractor model ready (MobileNetV2, output 10 classes)')
else:
    def run_demo():
        print('TensorFlow not available. Pseudocode:')
        print('- load pre-trained backbone (e.g., MobileNetV2)')
        print("- set base.trainable=False for feature extraction or True for fine-tuning")
        print("- add GlobalAveragePooling + Dense classifier head and compile")

if __name__ == '__main__':
    run_demo()

    # Exercises:
    # - Try feature extraction on a small dataset (freeze base, train head for few epochs).
    # - Then unfreeze last block and fine-tune with a smaller LR.
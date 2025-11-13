"""Day 24.07 — Local webcam inference stub (OpenCV optional)
Run time: ~12 minutes

- Captures frames from webcam and runs a dummy model prediction (replace with real model)
- If OpenCV is not installed, prints instructions to run later.
"""

import os
if os.getenv("SMOKE_TEST") == "1":
    print("SMOKE: skipping OpenCV webcam demo (no camera in CI)")
    raise SystemExit(0)

try:
    import cv2
    has_cv2 = True
except Exception:
    has_cv2 = False

import numpy as np


def dummy_predict(frame):
    # simple heuristic: mean brightness -> predict class 0/1
    m = frame.mean()
    return int(m > 127)

if __name__ == '__main__':
    if not has_cv2:
        print('OpenCV not available. Install opencv-python to run webcam demo: pip install opencv-python')
    else:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print('Webcam not available')
        else:
            print('Press q to quit. Running 50 frames...')
            for _ in range(50):
                ret, frame = cap.read()
                if not ret:
                    break
                # resize & gray for dummy predict
                small = cv2.resize(frame, (64, 64))
                pred = dummy_predict(small)
                cv2.putText(frame, f'pred: {pred}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                cv2.imshow('demo', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            cap.release()
            cv2.destroyAllWindows()

    # Exercises:
    # - Replace dummy_predict with a real model (onnx/tflite) loaded and run per-frame inference.
    # - Measure frame processing time with time.perf_counter.
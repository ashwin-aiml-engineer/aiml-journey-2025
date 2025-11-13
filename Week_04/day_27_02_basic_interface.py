"""Day 27.02 — Basic Gradio interface (text -> label)
Run time: ~10-15 minutes

- Minimal example: sentiment or echo demo with safe fallback if gradio not installed
"""

import os
if os.getenv("SMOKE_TEST") == "1":
    print("SMOKE: skipping Gradio basic interface demo")
    raise SystemExit(0)

try:
    import gradio as gr
    has_gradio = True
except Exception:
    has_gradio = False


def predict_text(text):
    # trivial heuristic: positive if contains 'good' or 'love'
    txt = text.lower()
    if 'good' in txt or 'love' in txt or 'great' in txt:
        return 'Positive'
    if 'bad' in txt or 'hate' in txt:
        return 'Negative'
    return 'Neutral'

if __name__ == '__main__':
    if not has_gradio:
        print('gradio not installed. Install with: pip install gradio')
        print("run demo after install: python Week_04\\day_27_02_basic_interface.py")
    else:
        iface = gr.Interface(fn=predict_text, inputs=gr.Textbox(lines=2, placeholder='Type here...'), outputs=gr.Label())
        iface.launch(share=False)

    # Exercises:
    # - Replace predict_text with a real pipeline (transformers) if available.
    # - Add example inputs and a title/description to the interface.
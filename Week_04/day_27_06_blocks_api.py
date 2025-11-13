"""Day 27.06 — Blocks API small layout: text + image tabs
Run time: ~15 minutes

- Demonstrates a compact Blocks layout with tabs and event handlers
"""

import os
if os.getenv("SMOKE_TEST") == "1":
    print("SMOKE: skipping Gradio Blocks demo")
    raise SystemExit(0)

try:
    import gradio as gr
    has_gradio = True
except Exception:
    has_gradio = False


def echo(text):
    return text[::-1]

if __name__ == '__main__':
    if not has_gradio:
        print('Gradio not installed. Install to run Blocks demo.')
    else:
        with gr.Blocks() as demo:
            with gr.Tab('Text'):
                txt = gr.Textbox(label='Enter text')
                out = gr.Textbox(label='Reversed')
                txt.change(fn=echo, inputs=txt, outputs=out)
            with gr.Tab('Image'):
                img = gr.Image()
                lbl = gr.Label()
        demo.launch(share=False)

    # Exercises:
    # - Add an examples table and a clear button to the Blocks demo.
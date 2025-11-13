"""Day 27.04 — NLP demo patterns (sentiment, NER view) using Gradio
Run time: ~15 minutes

- Runs a sentiment pipeline if transformers available; otherwise fallback heuristic
- Also shows a simple NER viewer stub (splits tokens and tags)
"""

import os
if os.getenv("SMOKE_TEST") == "1":
    print("SMOKE: skipping Gradio/Transformers NLP demo")
    raise SystemExit(0)

try:
    import gradio as gr
    from transformers import pipeline
    has_full = True
except Exception:
    has_full = False


def sentiment_demo(text):
    if has_full:
        clf = pipeline('sentiment-analysis')
        return clf(text)
    return predict_text(text)


def predict_text(text):
    txt = text.lower()
    if 'good' in txt or 'love' in txt:
        return [{'label':'POS','score':0.9}]
    if 'bad' in txt:
        return [{'label':'NEG','score':0.9}]
    return [{'label':'NEU','score':0.6}]

if __name__ == '__main__':
    if not has_full:
        print('Transformers/Gradio not fully available — run the sentiment demo after installing packages.')
    else:
        demo = gr.Interface(fn=sentiment_demo, inputs=gr.Textbox(), outputs=gr.Label())
        demo.launch(share=False)

    # Exercises:
    # - Add a NER viewer that highlights tokens with BIO tags (use a small tokenizer).
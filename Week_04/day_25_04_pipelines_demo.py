"""Day 25.04 — Pipelines demo: quick inference with transformers (safe stub)
Run time: ~15 minutes

- If `transformers` is installed, this runs a tiny pipeline example.
- Otherwise prints safe pseudocode and instructions.
"""

try:
    from transformers import pipeline
    has_hf = True
except Exception:
    has_hf = False

if has_hf:
    def run_demo():
        clf = pipeline('sentiment-analysis')
        print('Running sentiment pipeline on two examples:')
        print(clf('I love this course.'))
        print(clf('This is not good.'))
else:
    def run_demo():
        print('transformers not installed. To try pipeline, run: pip install transformers')
        print('Then run:')
        print("from transformers import pipeline\nclf = pipeline('sentiment-analysis')\nclf('text')")

if __name__ == '__main__':
    run_demo()

    # Exercises:
    # - Try a pipeline for 'question-answering' if you have transformers installed.
    # - Time how long the pipeline takes for a batch of 8 texts.
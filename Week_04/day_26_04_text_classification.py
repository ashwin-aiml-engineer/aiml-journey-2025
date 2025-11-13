"""Day 26.04 — Text classification fine-tune stub (BERT) or sklearn fallback
Run time: ~15 minutes

- If transformers available: show model/head replacement pseudocode
- Otherwise use sklearn pipeline on TF-IDF as quick runnable demo
"""

import os
if os.getenv("SMOKE_TEST") == "1":
    print("SMOKE: skipping Transformers text-classification demo")
    raise SystemExit(0)

try:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    import torch
    has_transformers = True
except Exception:
    has_transformers = False

if has_transformers:
    def run_demo():
        print('Transformers available: pseudocode to fine-tune BERT')
        print("tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')")
        print("model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)")
        print("# freeze backbone example: for p in model.base_model.parameters(): p.requires_grad=False")
else:
    def run_demo():
        print('Transformers not installed — running sklearn TF-IDF + LogisticRegression demo')
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        X = ['good product', 'bad experience', 'love this', 'not great']
        y = [1, 0, 1, 0]
        v = TfidfVectorizer().fit_transform(X)
        clf = LogisticRegression().fit(v, y)
        print('Toy sklearn accuracy:', clf.score(v, y))

if __name__ == '__main__':
    run_demo()

    # Exercises:
    # - If TF installed, replace head with new classification head and train for 1 epoch.
    # - With sklearn demo: try class imbalance by duplicating negative samples.
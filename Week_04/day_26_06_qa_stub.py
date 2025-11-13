"""Day 26.06 — Question Answering fine-tune stub (SQuAD-style)
Run time: ~15 minutes

- Shows triplet creation (context, question, answer span) and simple EM/F1 sketch
"""


def exact_match(pred, gold):
    return int(pred.strip() == gold.strip())


def f1_score(pred, gold):
    p_tokens = pred.split()
    g_tokens = gold.split()
    common = set(p_tokens) & set(g_tokens)
    if not common:
        return 0.0
    prec = len(common) / len(p_tokens)
    rec = len(common) / len(g_tokens)
    return 2 * prec * rec / (prec + rec)

if __name__ == '__main__':
    context = 'Paris is the capital of France.'
    question = 'What is the capital of France?'
    gold = 'Paris'
    pred = 'Paris'
    print('EM:', exact_match(pred, gold), 'F1:', round(f1_score(pred, gold), 3))

    # Exercises:
    # - Build a tiny SQuAD-like example JSON with 2 contexts and compute EM/F1 on sample preds.
    # - Prepare token start/end spans for training examples.
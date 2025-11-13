"""Day 26.05 — NER fine-tuning stub: BIO tags and token classification
Run time: ~12 minutes

- Shows how to prepare token labels and a simple BIO example
"""


def bio_tags(tokens, entities):
    # tokens: list[str], entities: list of (start_idx, end_idx, label)
    tags = ['O'] * len(tokens)
    for s, e, lbl in entities:
        tags[s] = 'B-' + lbl
        for i in range(s+1, e):
            tags[i] = 'I-' + lbl
    return tags

if __name__ == '__main__':
    toks = ['John', 'lives', 'in', 'New', 'York']
    ents = [(0,1,'PER'), (3,5,'LOC')]
    print('Tokens:', toks)
    print('BIO tags:', bio_tags(toks, ents))

    # Exercises:
    # - Implement conversion from character spans to token spans using a tokenizer.
    # - Create a small dataset in CoNLL format and parse it.
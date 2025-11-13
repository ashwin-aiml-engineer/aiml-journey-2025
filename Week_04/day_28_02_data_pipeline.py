"""Day 28.02 — Data pipeline stub: ingest, validate, simple ETL
Run time: ~15 minutes

- Safe, runnable stub that shows ingest -> validate -> write steps (no external services required)
"""

import csv
import json
from pathlib import Path


def validate_row(row):
    # simple validation: required keys and non-empty values
    required = ['id', 'text']
    for k in required:
        if k not in row or row[k].strip() == '':
            return False
    return True


def etl(input_path, out_path):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    out = []
    with open(input_path, encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            if validate_row(r):
                # simple transform: lowercase text
                r['text'] = r['text'].lower()
                out.append(r)
    with open(out_path, 'w', encoding='utf-8') as fw:
        json.dump(out, fw, ensure_ascii=False, indent=2)
    print('Wrote', len(out), 'validated records to', out_path)

if __name__ == '__main__':
    # create toy CSV
    inp = 'data/sample_input.csv'
    Path('data').mkdir(exist_ok=True)
    with open(inp, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'text'])
        writer.writeheader()
        writer.writerow({'id': '1', 'text': 'Hello World'})
        writer.writerow({'id': '2', 'text': ''})
        writer.writerow({'id': '3', 'text': 'Good product'})

    etl(inp, 'data/validated.json')

    # Exercises:
    # - Extend validate_row with schema checks (length, numeric fields).
    # - Add simple batching to process large CSVs.
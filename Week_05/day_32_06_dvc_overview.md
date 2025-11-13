DVC overview — quick guide (10–15 minutes)

Core ideas
- DVC is a data versioning tool that works with Git. Use it to track large data files and pipelines.
- Basic commands:
  - `dvc init` — initialize DVC in repo
  - `dvc add data/raw.csv` — track a data file (creates .dvc file)
  - `dvc push` / `dvc pull` — interact with remote storage (S3/GCS)
  - `dvc repro` — run pipeline stages reproducibly

Mini task
- Draft the DVC commands you'd run to version a dataset and push to an S3 remote.

Notes
- DVC keeps code and data references separate: the Git history stores pointers to data rather than the data itself.
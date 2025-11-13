"""Day 25.07 — Hugging Face Hub integration demo (safe stub)
Run time: ~15 minutes

- Shows how to programmatically list model metadata via `huggingface_hub` if installed
- Otherwise prints curl examples to fetch model info
"""

try:
    from huggingface_hub import hf_hub_url, model_info
    has_hub = True
except Exception:
    has_hub = False

if has_hub:
    def run_demo():
        info = model_info('distilbert-base-uncased')
        print('Model id:', info.modelId if hasattr(info, 'modelId') else info.modelId)
        print('Tags:', info.tags)
        print('Downloads:', getattr(info, 'downloads', 'N/A'))
else:
    def run_demo():
        print('huggingface_hub not installed. Example curl to get model card:')
        print("curl -s https://huggingface.co/api/models/distilbert-base-uncased | jq .")

if __name__ == '__main__':
    run_demo()

    # Exercises:
    # - Use `huggingface_hub` to download a tokenizer or model weights and inspect files.
    # - Explore model card keys (license, tags, pipeline_tag).
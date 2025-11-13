# Day 27.08 — Sharing & Deployment (Gradio/Spaces) concise notes

Local
- Launch with `share=False` for local testing
- Use `uvicorn`/FastAPI for production wrappers

Temporary Public Share
- `demo.launch(share=True)` creates a temporary public URL (useful for quick client demos)

Hugging Face Spaces
- Deploy Gradio apps to Spaces (public or private) for long-lived demos
- Use `requirements.txt` to pin dependencies

Production
- Use a container (Docker) and reverse proxy to put Gradio behind authentication
- Track usage and add rate-limits for expensive demos

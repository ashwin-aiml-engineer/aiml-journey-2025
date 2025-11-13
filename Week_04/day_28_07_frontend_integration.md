# Day 28.07 — Frontend Integration (Gradio / Streamlit / React)

Quick tips
- Use Gradio for quick demos, Streamlit for data apps, React for production frontends.
- Frontend calls backend /predict endpoints; keep payload small (IDs or compressed inputs).

File uploads
- Accept files on frontend, upload to object storage or proxy to backend.
- Validate file types and sizes before sending to model service.

Realtime
- Use WebSockets for live updates or long-running jobs; otherwise use polling.

Exercise
- Sketch a minimal React fetch call to POST text to /api/v1/predict and display JSON result.
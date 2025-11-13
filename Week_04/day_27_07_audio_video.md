# Day 27.07 — Audio & Video with Gradio (concise)

Audio
- Use `gr.Audio` for input/output (waveform or file)
- For speech recognition, connect a speech-to-text pipeline

Video
- `gr.Video` accepts file or webcam streams
- For real-time processing, use a lightweight model and small frame sizes

Tip
- Provide example audio/video files to make demos accessible to clients without a mic/webcam.
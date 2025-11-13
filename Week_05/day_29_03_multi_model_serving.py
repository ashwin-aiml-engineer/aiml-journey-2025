"""Day 29.03 — Multi-model serving stub (FastAPI) — runnable safe stub
Run time: ~15 minutes

- Starts a simple FastAPI app if FastAPI is installed; otherwise prints pseudocode.
- Demonstrates dynamic routing to different models and a factory pattern sketch.
"""

import os
if os.getenv("SMOKE_TEST") == "1":
    print("SMOKE: skipping FastAPI multi-model serving stub")
    raise SystemExit(0)

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    import uvicorn
    has_fastapi = True
except Exception:
    has_fastapi = False

class PredictRequest(BaseModel):
    model_name: str
    input: str

# Simple in-memory factory (replace with dynamic loader)
class ModelFactory:
    def __init__(self):
        self._models = {}
    def register(self, name, fn):
        self._models[name] = fn
    def predict(self, name, x):
        if name not in self._models:
            raise KeyError(name)
        return self._models[name](x)

# Dummy models
def sentiment_fn(text):
    t = text.lower()
    return {'label': 'POS' if 'good' in t else 'NEG' if 'bad' in t else 'NEU'}

def echo_fn(text):
    return {'echo': text[::-1]}

factory = ModelFactory()
factory.register('sentiment', sentiment_fn)
factory.register('echo', echo_fn)

if __name__ == '__main__':
    if not has_fastapi:
        print('FastAPI not installed. To run the multi-model server:')
        print('pip install fastapi uvicorn pydantic')
        print('\nPseudocode:')
        print('- Create FastAPI app, on startup load models into a factory')
        print("- POST /predict -> {model_name, input} -> factory.predict(model_name, input)")
    else:
        app = FastAPI(title='Multi-model Server (stub)')

        @app.post('/predict')
        async def predict(req: PredictRequest):
            try:
                out = factory.predict(req.model_name, req.input)
                return {'status': 200, 'result': out}
            except KeyError:
                raise HTTPException(status_code=404, detail='Model not found')

        print('Starting FastAPI multi-model stub on http://127.0.0.1:8000')
        uvicorn.run(app, host='127.0.0.1', port=8000)

    # Exercises:
    # - Extend ModelFactory to support loading models by version: factory.register('v1', fn)
    # - Add a warm-up endpoint that runs a dummy input through loaded models.
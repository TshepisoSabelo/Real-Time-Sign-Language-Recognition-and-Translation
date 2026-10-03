import asyncio

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

app = FastAPI()
app.state.latest_prediction = {"letter": "None", "confidence": 0.0}


class PredictionUpdate(BaseModel):
    letter: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)


@app.get("/")
def root():
    return {"message": "Hello, World!"}

@app.post("/start")
def start(name: str):
    return {"message": f"Hello, {name}, we are live!"}


@app.post("/prediction")
async def update_prediction(prediction: PredictionUpdate):
    app.state.latest_prediction = {
        "letter": prediction.letter,
        "confidence": prediction.confidence,
    }
    return {"status": "ok"}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    last_prediction = None
    try:
        while True:
            prediction = app.state.latest_prediction
            current_prediction = (prediction["letter"], prediction["confidence"])
            if current_prediction != last_prediction:
                await websocket.send_json(prediction)
                last_prediction = current_prediction

            await asyncio.sleep(0.05)
    except WebSocketDisconnect:
        pass

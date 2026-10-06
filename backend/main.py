import asyncio
import logging
from contextlib import asynccontextmanager

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from ml.src.hand_detection.hand_detect import (
    DEFAULT_CHECKPOINT,
    DEFAULT_DETECTOR,
    OpenCVPreview,
    load_inference_models,
    process_frame_with_preview,
)

logger = logging.getLogger(__name__)
MAX_FRAME_BYTES = 4 * 1024 * 1024


@asynccontextmanager
async def lifespan(application: FastAPI):
    detector, classifier = load_inference_models(
        detector_path=DEFAULT_DETECTOR,
        checkpoint_path=DEFAULT_CHECKPOINT,
    )
    application.state.detector = detector
    application.state.classifier = classifier
    application.state.inference_lock = asyncio.Lock()
    application.state.preview = OpenCVPreview()
    application.state.recognition_sessions = 0
    try:
        yield
    finally:
        await run_in_threadpool(application.state.preview.close)


app = FastAPI(lifespan=lifespan)
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


@app.websocket("/ws/recognition")
async def recognition_websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    websocket.app.state.recognition_sessions += 1
    await run_in_threadpool(websocket.app.state.preview.start)
    try:
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                break

            frame_bytes = message.get("bytes")
            if not frame_bytes:
                await websocket.send_json({
                    "type": "error",
                    "status": "error",
                    "message": "Expected a non-empty binary JPEG frame.",
                })
                continue

            if len(frame_bytes) > MAX_FRAME_BYTES:
                await websocket.send_json({
                    "type": "error",
                    "status": "error",
                    "message": "Camera frame exceeds the maximum allowed size.",
                })
                continue

            try:
                image = cv2.imdecode(
                    np.frombuffer(frame_bytes, dtype=np.uint8),
                    cv2.IMREAD_COLOR,
                )
            except cv2.error:
                image = None

            if image is None:
                await websocket.send_json({
                    "type": "error",
                    "status": "error",
                    "message": "Could not decode the camera frame.",
                })
                continue

            if image.shape[0] * image.shape[1] > 2_000_000:
                await websocket.send_json({
                    "type": "error",
                    "status": "error",
                    "message": "Camera frame dimensions exceed the supported limit.",
                })
                continue

            try:
                async with websocket.app.state.inference_lock:
                    prediction, tracker_frame, hand_frame = await run_in_threadpool(
                        process_frame_with_preview,
                        image,
                        websocket.app.state.detector,
                        websocket.app.state.classifier,
                    )
                websocket.app.state.preview.publish(tracker_frame, hand_frame)
            except Exception:
                logger.exception("Sign recognition failed for a camera frame")
                await websocket.send_json({
                    "type": "error",
                    "status": "error",
                    "message": "The backend could not process this camera frame.",
                })
                continue

            no_hand_detected = prediction["letter"] == "None"
            await websocket.send_json({
                "type": "prediction",
                **prediction,
                "status": "no_hand" if no_hand_detected else "success",
            })
    except WebSocketDisconnect:
        logger.debug("Recognition WebSocket disconnected")
    finally:
        websocket.app.state.recognition_sessions -= 1
        if websocket.app.state.recognition_sessions == 0:
            await run_in_threadpool(websocket.app.state.preview.close)

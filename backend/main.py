from fastapi import FastAPI
from ml.src.hand_detection import hand_detection
app = FastAPI()

@app.get("/")
def root():
    return {"message": "Hello, World!"}

@app.post("/start")
def start(name: str):
    return {"message": f"Hello, {name}, we are live!"}
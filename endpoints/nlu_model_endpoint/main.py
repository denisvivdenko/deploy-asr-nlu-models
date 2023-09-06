import uvicorn
from fastapi import (
    FastAPI, 
    UploadFile, 
    HTTPException
)
from src.utils.inference_onnx import NLUONNXInference
from src.utils.configuration import load_params
import requests
import os

app = FastAPI()
params = load_params("params.yaml")
model = NLUONNXInference(
    model_name=params["convert_nlu_model"]["input_model_path_or_id"],
    onnx_path=params["convert_nlu_model"]["output_model_path"]
)

@app.get("/")
def read_root():
    return "This is NLU model endpoint"

@app.post("/predict")
async def predict(file: UploadFile = None):
    if not file:
        raise HTTPException(status_code=400, detail="File is not provided")

    file_extension = file.filename.split(".")[-1]

    if file_extension not in ["wav", "mp3", "flac"]:
        raise HTTPException(status_code=400, detail="Invalid file format")

    contents = await file.read()
    response = requests.post(
        os.getenv("ASR_MODEL_ENDPOINT", None), 
        files={"file": (file.filename, contents)}
    )
    transcript = response.json()["transcript"]

    prediction = model.predict(transcript)
    return {"transcript": transcript , "intent": prediction}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
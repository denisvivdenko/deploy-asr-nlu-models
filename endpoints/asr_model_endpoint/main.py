import uvicorn
from fastapi import (
    FastAPI, 
    UploadFile, 
    HTTPException
)
from src.utils.inference_onnx import Wav2Vec2ONNXInference
from src.utils.configuration import load_params

app = FastAPI()
params = load_params("params.yaml")
model = Wav2Vec2ONNXInference(
    model_name=params["convert_asr_model"]["input_model_path_or_id"],
    onnx_path=params["convert_asr_model"]["output_model_path"]
)

@app.get("/")
def read_root():
    return "This is ASR model endpoint"

@app.post("/predict")
def predict(file: UploadFile = None):
    if not file:
        raise HTTPException(status_code=400, detail="File is not provided")

    file_extension = file.filename.split(".")[-1]

    if file_extension not in ["wav", "mp3", "flac"]:
        raise HTTPException(status_code=400, detail="Invalid file format")

    prediction = model.predict(file.file)
    return {"prediction": prediction}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
import soundfile as sf
import torch
import onnxruntime as rt
import numpy as np

from transformers import Wav2Vec2Processor, AutoTokenizer

# Improvements: 
# - gpu / cpu flag
# - convert non 16 khz sample rates
# - inference time log

class Wav2Vec2ONNXInference():
    def __init__(self,model_name,onnx_path):
        self.processor = Wav2Vec2Processor.from_pretrained(model_name) 
        options = rt.SessionOptions()
        options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.model = rt.InferenceSession(onnx_path, options)

    def buffer_to_text(self, audio_buffer):
        if(len(audio_buffer)==0):
            return ""

        inputs = self.processor(torch.tensor(audio_buffer), sampling_rate=16_000, return_tensors="np", padding=True)

        input_values = inputs.input_values
        onnx_outputs = self.model.run(None, {self.model.get_inputs()[0].name: input_values})[0]
        prediction = np.argmax(onnx_outputs, axis=-1)

        transcription = self.processor.decode(prediction.squeeze().tolist())
        return transcription.lower()

    def predict(self, filename: str):
        audio_input, samplerate = sf.read(filename)
        assert samplerate == 16000
        return self.buffer_to_text(audio_input)
    

class NLUONNXInference():
    def __init__(self,model_name,onnx_path):
        self.processor = AutoTokenizer.from_pretrained(model_name)
        options = rt.SessionOptions()
        options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.model = rt.InferenceSession(onnx_path, options)

    def predict(self, text):
        inputs = self.processor(text, return_tensors="np", max_length=512, truncation=True, padding="max_length")
        onnx_inputs = {name: inputs[name] for name in ["input_ids", "attention_mask"]}

        onnx_outputs = self.model.run(None, onnx_inputs)[0]
        prediction = np.argmax(onnx_outputs, axis=-1)
        return prediction


if __name__ == "__main__":
    print("ASR test")
    asr = Wave2Vec2ONNXInference(
        "facebook/wav2vec2-base-960h",
        "models/wav2vec2-base-960h.onnx"
    )
    prediction = asr.predict("data/test.wav")
    print(prediction)

    print("NLU test")
    nlu = NLUONNXInference(
        "sankar1535/slurp-intent_baseline-distilbert-base-uncased",
        "models/slurp-intent_baseline-distilbert-base-uncased.onnx"
    )
    prediction = nlu.predict("Hello, world!")
    print(prediction)
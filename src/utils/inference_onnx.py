import soundfile as sf
import torch
import onnxruntime as rt
from typing import BinaryIO
import numpy as np
from transformers import Wav2Vec2Processor, AutoTokenizer, AutoModelForSequenceClassification

from src.utils.configuration import logging

class Wav2Vec2ONNXInference():
    def __init__(self, model_name: str, onnx_path: str) -> None:
        """
        Args:
            model_name (str): The name or path of the pretrained model.
            onnx_path (str): The path to the ONNX model.
        """
        self.processor = Wav2Vec2Processor.from_pretrained(model_name) 
        options = rt.SessionOptions()
        options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.model = rt.InferenceSession(onnx_path, options)

    def buffer_to_text(self, audio_buffer: np.ndarray) -> str:
        """Convert audio buffer to text using ASR model.

        Args:
            audio_buffer (np.ndarray): The audio buffer.

        Returns:
            str: The transcribed text.
        """
        if not list(audio_buffer): raise Exception("Audio buffer is empty")

        inputs = self.processor(torch.tensor(audio_buffer), sampling_rate=16_000, return_tensors="np", padding=True)

        input_values = inputs.input_values
        onnx_outputs = self.model.run(None, {self.model.get_inputs()[0].name: input_values})[0]
        prediction = np.argmax(onnx_outputs, axis=-1)

        transcription = self.processor.decode(prediction.squeeze().tolist())
        return transcription.lower()

    def predict(self, filename: str | BinaryIO) -> str:
        """Predict the transcription of the audio file.

        Args:
            filename (Union[str, BinaryIO]): The path or BinaryIO object representing the audio file.

        Returns:
            str: The transcription.
        """
        audio_input, samplerate = sf.read(filename)
        assert samplerate == 16000
        return self.buffer_to_text(audio_input)
    

class NLUONNXInference():
    def __init__(self, model_name, onnx_path):
        """Initialize the NLU model for inference.

        Args:
            model_name (str): The name or path of the pretrained model.
            onnx_path (str): The path to the ONNX model.
        """
        self.processor = AutoTokenizer.from_pretrained(model_name)
        options = rt.SessionOptions()
        options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
        model = AutoModelForSequenceClassification.from_pretrained("sankar1535/slurp-intent_baseline-distilbert-base-uncased")
        self.id2label = model.config.id2label
        del model

        self.model = rt.InferenceSession(onnx_path, options)
        
    def predict(self, text: str) -> str:
        """Predict the intent of the text using NLU model.

        Args:
            text (str): The text to predict the intent for.

        Returns:
            str: The predicted intent.
        """
        inputs = self.processor(text, return_tensors="np", max_length=512, truncation=True, padding="max_length")
        onnx_inputs = {name: inputs[name] for name in ["input_ids", "attention_mask"]}

        onnx_outputs = self.model.run(None, onnx_inputs)[0]
        predicted_id = np.argmax(onnx_outputs)

        if self.id2label:
            return self.id2label[predicted_id]
        
        raise Exception(f"Unknown label for this id: {predicted_id}.")


if __name__ == "__main__":
    logging.info("ASR test")
    asr = Wav2Vec2ONNXInference(
        "facebook/wav2vec2-base-960h",
        "models/wav2vec2-base-960h.onnx"
    )
    prediction = asr.predict("data/test.wav")
    logging.info(prediction)

    logging.info("NLU test")
    nlu = NLUONNXInference(
        "sankar1535/slurp-intent_baseline-distilbert-base-uncased",
        "models/slurp-intent_baseline-distilbert-base-uncased.onnx"
    )
    prediction = nlu.predict("Hello, world!")
    logging.info(prediction)
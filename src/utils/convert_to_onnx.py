import torch
import argparse
from pathlib import Path

from transformers import (
    Wav2Vec2ForCTC, 
    AutoTokenizer, 
    AutoModelForSequenceClassification
)
from src.utils.configuration import logging


def convert_wav2vec2_model2onnx(model_id_or_path: str, save_path: str) -> None:
    """Convert a Wav2Vec2 model from HuggingFace to ONNX format."""
    logging.info(f"Converting {model_id_or_path} to onnx")
    model = Wav2Vec2ForCTC.from_pretrained(model_id_or_path)
    audio_len = 250000

    x = torch.randn(1, audio_len, requires_grad=True)

    torch.onnx.export(model,                        # model being run
                    x,                              # model input (or a tuple for multiple inputs)
                    save_path,                      # where to save the model (can be a file or file-like object)
                    export_params=True,             # store the trained parameter weights inside the model file
                    opset_version=11,               # the ONNX version to export the model to
                    do_constant_folding=True,       # whether to execute constant folding for optimization
                    input_names = ['input'],        # the model's input names
                    output_names = ['output'],      # the model's output names
                    dynamic_axes={'input' : {1 : 'audio_len'},    # variable length axes
                                'output' : {1 : 'audio_len'}})


def convert_nlu_model2onnx(model_id_or_path: str, save_path: str) -> None:
    """Convert an NLU model from HuggingFace to ONNX format."""
    logging.info(f"Converting {model_id_or_path} to onnx")
    tokenizer = AutoTokenizer.from_pretrained(model_id_or_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_id_or_path)
    model.eval()
    inputs = tokenizer("Hello, world!", return_tensors="pt")
    dummy_input = {key: value for key, value in inputs.items()}

    torch.onnx.export(
        model,
        dummy_input,
        save_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["output"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "output": {0: "batch_size"}
        },
        opset_version=11
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--asr-model",
        type=str,
        default="facebook/wav2vec2-base-960h",
        help="Model HuggingFace ID or path that will converted to ONNX",
    )
    parser.add_argument(
        "--nlu-model",
        type=str,
        default="sankar1535/slurp-intent_baseline-distilbert-base-uncased",
        help="Model HuggingFace ID or path that will converted to ONNX",
    )
    args = parser.parse_args()

    asr_onnx_save_fpath = Path("models") / (args.asr_model.split("/")[-1] + ".onnx")
    nlu_onnx_save_fpath = Path("models") / (args.nlu_model.split("/")[-1] + ".onnx")
    convert_wav2vec2_model2onnx(args.asr_model, asr_onnx_save_fpath)
    convert_nlu_model2onnx(args.nlu_model, nlu_onnx_save_fpath)

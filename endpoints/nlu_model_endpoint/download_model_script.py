from src.utils.inference_onnx import NLUONNXInference
from src.utils.configuration import load_params


if __name__ == "__main__":
    params = load_params("params.yaml")
    model = NLUONNXInference(
        model_name=params["convert_nlu_model"]["input_model_path_or_id"],
        onnx_path=params["convert_nlu_model"]["output_model_path"]
    )

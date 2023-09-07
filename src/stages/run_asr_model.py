import argparse
import pandas as pd
from tqdm import tqdm

from src.utils.configuration import load_params, logging
from src.utils.inference_onnx import Wav2Vec2ONNXInference


if __name__ == "__main__":
    logging.info("Run stage ASR model...")
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--config",
        dest="config",
        type=str,
        required=True
    )
    args = argparser.parse_args()
    params = load_params(args.config)

    model = Wav2Vec2ONNXInference(
        model_name=params["convert_asr_model"]["input_model_path_or_id"],
        onnx_path=params["convert_asr_model"]["output_model_path"]
    )
    logging.info(f"Loaded model. {params['convert_asr_model']['input_model_path_or_id']}. {params['convert_asr_model']['output_model_path']}")

    data = pd.read_csv(params["data_preprocessing"]["output_fpath"])
    logging.info(f"Dataset. {data.info()}\n{data.head(5)}")

    logging.info("Start inference...")
    predictions = {"slurp_id": [], "transcript": []}
    for _, row in tqdm(data.iterrows()):
        try:
            prediction = model.predict(row["recordings"])
        except Exception as e:
            logging.error(f"Error: failed to predict. {e}")
            continue
        logging.info(f"\nReal: {row['sentence']}\nPred: {prediction}\n")
        predictions["slurp_id"].append(row["slurp_id"])
        predictions["transcript"].append(prediction)
    
    predictions = pd.DataFrame(predictions)
    predictions.to_csv(params["asr_inference"]["output_fpath"])
    logging.info(f"Saved predictions to {params['asr_inference']['output_fpath']}")

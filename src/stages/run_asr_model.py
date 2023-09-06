import argparse
import logging
from tqdm import tqdm

from src.utils.configuration import load_params
from src.utils.inference_onnx import Wav2Vec2ONNXInference
from src.utils.data_preprocessing import preprocess_slurp_dataset

if __name__ == "__main__":
    logging.info("Run stage ASR model...")
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--config",
        dest="config",
        type=str
    )
    args = argparser.parse_args()
    params = load_params(args.config)

    model = Wav2Vec2ONNXInference(
        model_name=params["convert_asr_model"]["input_model_path_or_id"],
        onnx_path=params["convert_asr_model"]["output_model_path"]
    )
    logging.info(f"Loaded model. {params['convert_asr_model']['input_model_path_or_id']}. {params['convert_asr_model']['output_model_path']}")

    data = preprocess_slurp_dataset(
        recordings_metadata_fpath=params["dataset"]["dev_recordings_metadata_fpath"],
        recordings_dir=params["dataset"]["recordings_dir"]
    )
    logging.info(f"Prepared dataset. {data.info()}\n{data.head(5)}")

    logging.info("Start inference...")
    for _, row in tqdm(data.iterrows()):
        prediction = model.predict(row["recordings"])
        logging.info(f"Real: {row['sentence']}\t Pred: {prediction}")

import argparse
from tqdm import tqdm
import pandas as pd

from src.utils.configuration import load_params, logging
from src.utils.inference_onnx import NLUONNXInference


if __name__ == "__main__":
    logging.info("Run stage NLU model...")
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--config",
        dest="config",
        type=str,
        required=True
    )
    args = argparser.parse_args()
    params = load_params(args.config)

    model = NLUONNXInference(
        model_name=params["convert_nlu_model"]["input_model_path_or_id"],
        onnx_path=params["convert_nlu_model"]["output_model_path"]
    )
    logging.info(f"Loaded model. {params['convert_nlu_model']['input_model_path_or_id']}. {params['convert_nlu_model']['output_model_path']}")

    data = pd.read_csv(params["data_preprocessing"]["output_fpath"])
    logging.info(f"Dataset. {data.info()}\n{data.head(5)}")

    asr_predictions = pd.read_csv(params["asr_inference"]["output_fpath"])

    data = data.merge(asr_predictions, on="slurp_id", how="right")

    logging.info("Start inference...")
    predictions = {"slurp_id": [], "asr_predicted_intent": [], "groundtruth_predicted_intent": []}
    for _, row in tqdm(data.iterrows()):
        asr_prediction = model.predict(row["transcript"])
        groundtruth_prediction = model.predict(row["sentence"])
        logging.info(f"\nReal: {row['intent']}\nASR Pred: {asr_prediction}\nGroundtruth Pred: {groundtruth_prediction}\n")
        predictions["slurp_id"].append(row["slurp_id"])
        predictions["asr_predicted_intent"].append(asr_prediction)
        predictions["groundtruth_predicted_intent"].append(groundtruth_prediction)
    predictions = pd.DataFrame(predictions)
    predictions.to_csv(params["nlu_inference"]["output_fpath"])
    logging.info(f"Finished infrence. Saved results to {params['nlu_inference']['output_fpath']}")
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
    predictions = {"slurp_id": [], "predictions": []}
    for _, row in tqdm(data.iterrows()):
        prediction = model.predict(row["recordings"])
        logging.info(f"\nReal: {row['intent']}\nPred: {prediction}\n")
        predictions["slurp_id"].append(row["slurp_id"])
        predictions["predictions"].append(prediction)
    predictions = pd.DataFrame(predictions)
    predictions.to_csv(params["nlu_inference"]["output_fpath"])
    logging.info(f"Finished infrence. Saved results to {params['nlu_inference']['output_fpath']}")
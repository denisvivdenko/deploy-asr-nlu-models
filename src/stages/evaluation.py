import argparse
import json
import pandas as pd

from src.utils.configuration import load_params, logging


def compute_accuracy(y_true, y_pred):
    assert len(y_true) == len(y_pred), "Both lists/series must have the same length"
    correct = sum(t == p for t, p in zip(y_true, y_pred))
    total = len(y_true)
    return correct / total


if __name__ == "__main__":
    logging.info("Run stage Evaluation...")
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--config",
        dest="config",
        type=str,
        required=True
    )
    args = argparser.parse_args()
    params = load_params(args.config)

    true_values = pd.read_csv(params["data_preprocessing"]["output_fpath"])
    predicted_values = pd.read_csv(params["nlu_inference"]["output_fpath"])

    true_values_vs_predicted = true_values.merge(predicted_values, on="slurp_id", how="left")
    metric = compute_accuracy(true_values_vs_predicted['intent'], true_values_vs_predicted['predictions'])
    logging.info(f"Accuracy: {metric}")

    with open("metrics/metrics.json", "w") as f:
        json.dump({"accuracy": metric}, f)
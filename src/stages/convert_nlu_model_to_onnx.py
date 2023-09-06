import argparse

from src.utils.configuration import load_params, logging
from src.utils.convert_to_onnx import convert_nlu_model2onnx


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--config",
        dest="config",
        type=str,
        required=True
    )
    args = argparser.parse_args()
    config_fpath = args.config

    params = load_params(config_fpath)

    logging.info(f"""Start converting nlu model to onnx.\
    Model {params['convert_asr_model']['input_model_path_or_id']}
    will be saved to {params['convert_asr_model']['output_model_path']}
    """)
    convert_nlu_model2onnx(
        params["convert_nlu_model"]["input_model_path_or_id"],
        params["convert_nlu_model"]["output_model_path"]
    )
    logging.info("Finished converting nlu model to onnx")

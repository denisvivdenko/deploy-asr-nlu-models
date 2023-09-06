import argparse

from src.utils.configuration import load_params, logging
from src.utils.data_preprocessing import preprocess_slurp_dataset


if __name__ == "__main__":
    logging.info("Run dataset preprocessing stage...")
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--config",
        dest="config",
        type=str,
        required=True
    )
    args = argparser.parse_args()
    params = load_params(args.config)

    data = preprocess_slurp_dataset(
        recordings_metadata_fpath=params["data_preprocessing"]["dev_recordings_metadata_fpath"],
        recordings_dir=params["data_preprocessing"]["recordings_dir"]
    )
    data = data.sample(
        n=int(data.shape[0] * params["base"]["sample_size"]), 
        random_state=params["base"]["random_state"]
    ).reset_index(drop=True)
    data.to_csv(params["data_preprocessing"]["output_fpath"])
    logging.info(f"Prepared and saved dataset to {params['data_preprocessing']['output_fpath']}. {data.info()}\n{data.head(5)}")

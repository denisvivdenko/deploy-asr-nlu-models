import pandas as pd
from pathlib import Path
import os
import tarfile

from src.utils.configuration import logging


def unpack_dataset(tar_path: str, extract_to: str = None) -> None:
    """
    Unpacks a .tar.gz file.

    Parameters:
    - tar_path (str): Path to the .tar.gz file.
    - extract_to (str, optional): Directory to extract files to. If None, extracts to the same directory as the .tar.gz file.
    """
    if extract_to is None:
        extract_to = os.path.dirname(tar_path)

    if not os.path.exists(extract_to):
        os.makedirs(extract_to)

    with tarfile.open(tar_path, 'r:gz') as tar_ref:
        tar_ref.extractall(extract_to)


def filter_best_recording(recordings: list[dict]) -> str:
    """
    Filters recordings to find the best one based on WER (Word Error Rate).
    
    Parameters:
    - recordings (list[dict]): A list of dictionaries containing recording details.

    Returns:
    - str: The file path of the best recording. Returns None if no suitable recording is found.
    """
    headset_subset = list(filter(lambda x: "headset" in x["file"], recordings))
    if not headset_subset:
        return None
    best_recording = min(headset_subset, key=lambda x: x["wer"])
    return best_recording["file"]


def preprocess_slurp_dataset(recordings_metadata_fpath: str, recordings_dir: str) -> pd.DataFrame:
    """
    Preprocesses the SLURP dataset based on metadata and recordings directory.

    Parameters:
    - recordings_metadata_fpath (str): File path to the recordings metadata JSONL file.
    - recordings_dir (str): The directory where the recordings are stored.

    Returns:
    - pd.DataFrame: A pandas DataFrame containing preprocessed data.
    """
    data = pd.read_json(
        open(recordings_metadata_fpath, "r"), 
        lines=True
    )
    data = data[["slurp_id", "sentence", "intent", "recordings"]]
    data.recordings = data.recordings.apply(filter_best_recording)
    data = data.dropna()
    data.recordings = data.recordings.apply(lambda x: Path(recordings_dir) / x)
    return data


def main():
    recordings_metadata_fpath = "data/slurp_dataset/slurp/devel.jsonl"
    recordings_dir = "data/slurp_dataset/audio/slurp_real"
    
    processed_data = preprocess_slurp_dataset(recordings_metadata_fpath, recordings_dir)
    logging.info(processed_data)

if __name__ == "__main__":
    main()
import yaml


def load_params(params_fpath: str) -> dict:
    """
    params_fpath: str (file path to yaml file)
    return: dict
    """
    with open(params_fpath, "r") as file:
        return yaml.safe_load(file)

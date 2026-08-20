import yaml

def load_config(path: str):
    with open(path, "r") as f:
        args = yaml.safe_load(f)
    return args

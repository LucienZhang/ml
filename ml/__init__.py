from pathlib import Path


def get_base_dir():
    return Path(__file__).resolve().parent


def get_model_dir():
    model_dir = get_base_dir() / 'weights'
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir


def get_log_dir(model_name, experiment_name):
    if experiment_name:
        model_name += '_' + experiment_name
    log_dir = get_base_dir() / 'logs' / model_name
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir

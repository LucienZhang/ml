from pathlib import Path


def get_model_log_dir(model_name, experiment_name):
    base_path = Path(__file__).resolve().parent
    model_dir = base_path / 'models'
    if experiment_name:
        model_name += '_' + experiment_name
    log_dir = base_path / 'logs' / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    return model_dir, log_dir

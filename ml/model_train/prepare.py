from pathlib import Path


def get_model_log_dir(model_name):
    base_path = Path(__file__).resolve().parent
    model_dir = base_path / 'models'
    log_dir = base_path / 'logs' / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    return model_dir, log_dir

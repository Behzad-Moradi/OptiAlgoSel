from API.config import MODEL_PATH
from API.services.load_model_registery import load_model_registry

def get_production_model():

    model_registry = load_model_registry()
    production_model = next(
        (model for model in model_registry if model["status"] == "production"),
        None
    )

    if production_model is None:
        return None

    production_model["path"] = (
        MODEL_PATH / production_model["file"]
    )

    return production_model

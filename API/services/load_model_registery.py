import json
from API.config import settings

def load_model_registry():
    with open(settings.MODEL_REGISTRY, "r") as f:
        return json.load(f)
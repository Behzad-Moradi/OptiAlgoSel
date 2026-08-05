import json
from API.config import MODEL_REGISTRY

def load_model_registry():
    with open(MODEL_REGISTRY, "r") as f:
        return json.load(f)
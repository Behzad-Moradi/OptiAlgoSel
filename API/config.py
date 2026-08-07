from pathlib import Path
from dotenv import load_dotenv
import os

BASE_DIR = Path(__file__).resolve().parent.parent

ENV_PATH = BASE_DIR / ".env"

load_dotenv(ENV_PATH)

DATABASE_PATH = Path(
    os.getenv(
        "DATABASE_PATH",
        "DataBase/optialgosel.db"
    )
)


MODEL_PATH = Path(
    os.getenv(
        "MODEL_PATH",
        "TrainedModels"
    )
)


MODEL_REGISTRY = Path(
    os.getenv(
        "MODEL_REGISTRY",
        "TrainedModels/model_registry.json"
    )
)


SENDER_EMAIL = os.getenv("SENDER_EMAIL")

APP_PASSWORD = os.getenv("APP_PASSWORD")
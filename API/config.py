from pathlib import Path
from pydantic import field_validator, EmailStr
from pydantic_settings import BaseSettings, SettingsConfigDict

# Get your base directory just like before
BASE_DIR = Path(__file__).resolve().parent.parent

class Settings(BaseSettings):
    
    
    
    # 1. Define fields and types (Missing fields will automatically throw a CRITICAL error)
    DB_HOST: str = "localhost"
    DB_PORT: int = 5432
    DB_NAME: str
    DB_USER: str
    DB_PASSWORD: str
    
    DATABASE_PATH: Path  
    MODEL_PATH: Path
    MODEL_REGISTRY: Path
    SENDER_EMAIL: EmailStr
    APP_PASSWORD: str

    # 2. Tell Pydantic to read from your .env file
    model_config = SettingsConfigDict(env_file=BASE_DIR / ".env", extra="ignore")

    # 3. Automatically turn relative paths into absolute paths
    @field_validator("DATABASE_PATH", "MODEL_PATH", "MODEL_REGISTRY", mode="before")
    @classmethod
    def make_paths_absolute(cls, value: str) -> Path:
        path_obj = Path(value)
        if not path_obj.is_absolute():
            return BASE_DIR / path_obj
        return path_obj

# Instantiate the settings. 
# This single line handles loading, validating, and path conversion.
settings = Settings()

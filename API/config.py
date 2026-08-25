from pathlib import Path
from pydantic import field_validator, EmailStr
from pydantic_settings import BaseSettings, SettingsConfigDict
from Utils.secrets import get_rds_credentials

# Get your base directory just like before
BASE_DIR = Path(__file__).resolve().parent.parent

class Settings(BaseSettings):

    # Define fields and types (Missing fields will automatically throw a CRITICAL error)
    DB_HOST: str = "localhost"
    DB_PORT: int = 5432
    DB_NAME: str
    DB_USER: str = ""
    DB_PASSWORD: str = ""
    
    DATABASE_PATH: Path  
    MODEL_PATH: Path
    MODEL_REGISTRY: Path
    SENDER_EMAIL: EmailStr
    APP_PASSWORD: str
    ENVIRONMENT: str = "development"

    # Tell Pydantic to read from your .env file
    model_config = SettingsConfigDict(env_file=BASE_DIR / ".env", extra="ignore")

    # Automatically turn relative paths into absolute paths
    @field_validator("DATABASE_PATH", "MODEL_PATH", "MODEL_REGISTRY", mode="before")
    @classmethod
    def make_paths_absolute(cls, value: str) -> Path:
        path_obj = Path(value)
        if not path_obj.is_absolute():
            return BASE_DIR / path_obj
        return path_obj
    # Load RDS credentials from AWS Secrets Manager when running in production.
    def model_post_init(self, __context):
        if self.ENVIRONMENT == "production":
            credentials = get_rds_credentials()
            self.DB_USER = credentials["username"]
            self.DB_PASSWORD = credentials["password"]

# Instantiate the settings. 
# This single line handles loading, validating, and path conversion.
settings = Settings()

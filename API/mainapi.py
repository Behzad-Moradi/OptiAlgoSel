from fastapi import FastAPI
import uvicorn
import logging
from API.include_routers import include_routers
from API.logging_config import setup_logging
#from services.database_service import create_database_schema

DATABASE_DIR ='DataBase/optialgosel.db'

setup_logging()

logger = logging.getLogger(__name__)

app = FastAPI(title="OptiAlgoSel API", description="This is the API for OptiAlgoSel.", version="1.0.0")
logger.info("OptiAlgoSel API started")
include_routers(app)

if __name__ == "__main__":

    #create_database_schema(DATABASE_DIR)
    uvicorn.run("mainapi:app", host="127.0.0.1", port=8000, reload=True)
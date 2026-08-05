from fastapi import FastAPI
import uvicorn
from include_routers import include_routers
from services.database_service import create_database_schema

DATABASE_DIR ='DataBase/optialgosel.db'


app = FastAPI(title="OptiAlgoSel API", description="This is the API for OptiAlgoSel.", version="1.0.0")
include_routers(app)

if __name__ == "__main__":

    #create_database_schema(DATABASE_DIR)
    uvicorn.run("mainapi:app", host="127.0.0.1", port=8000, reload=True)
from fastapi import APIRouter, Depends
import psycopg
import logging
from API.services.connect_database_pg import get_db_pg

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/algorithms", tags=["Optimisation Algorithms"])

@router.get("/", description="This endpoint retrieves the list of optimisation algorithms available in the system.", summary="List of optimisation algorithms.")
def get_algorithms(conn: psycopg.Connection = Depends(get_db_pg)):
    logger.info("Retrieving optimisation algorithm portfolio request received")
    
    with conn.cursor() as cur:
        cur.execute("SELECT algorithm_name, algorithm_description FROM algorithms")
        algorithms = cur.fetchall()

    logger.info("Optimisation algorithm portfolio retrieved")
    return {"Optimisation Algorithm Portfolio": algorithms}
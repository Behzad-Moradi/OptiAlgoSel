from fastapi import APIRouter, Depends
import sqlite3
from API.services.connect_database import get_db


router = APIRouter(prefix="/algorithms", tags=["Optimisation Algorithms"])

@router.get("/", description="This endpoint retrieves the list of optimisation algorithms available in the system.", summary="List of optimisation algorithms.")
async def get_algorithms(conn: sqlite3.Connection = Depends(get_db)):
    cur = conn.cursor()
    cur.execute("SELECT algorithm_name, algorithm_description FROM algorithms")
    algorithms = cur.fetchall()
    cur.close()
    return {"Optimisation Algorithm Portfolio": algorithms}
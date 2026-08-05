from fastapi import APIRouter, Path, Depends, status
import sqlite3
from API.services.connect_database import get_db

router = APIRouter(prefix="/result", tags=["Result"])

@router.get("/{request_d}", status_code=status.HTTP_200_OK, description="This endpoint retrieves the result of a prediction request.", summary="Result of a prediction request.")
async def get_result(request_d: int=Path(..., description="The request id of the prediction request."), conn: sqlite3.Connection = Depends(get_db)):
    cur = conn.cursor()
    cur.execute("SELECT predicted_algorithms FROM results WHERE request_id = ?", (request_d,))
    result = cur.fetchone()
    cur.close()
    return result[0]
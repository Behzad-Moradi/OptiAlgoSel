from fastapi import APIRouter, Path, Depends, status
import psycopg
from API.services.connect_database_pg import get_db_pg
from fastapi import HTTPException

router = APIRouter(prefix="/result", tags=["Result"])

@router.get("/{request_id}", status_code=status.HTTP_200_OK, description="This endpoint retrieves the result of a prediction request.", summary="Result of a prediction request.")
def get_result(request_id: int=Path(..., description="The request id of the prediction request."), conn: psycopg.Connection = Depends(get_db_pg)):
    
    with conn.cursor() as cur:
        cur.execute("SELECT predicted_algorithms FROM results WHERE request_id = %s", (request_id,))
        result = cur.fetchone()
    
    if result is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"No result found for request ID {request_id}.")

    return {"request_id": request_id, "predicted_algorithms": result[0]}

from fastapi import APIRouter, HTTPException, Depends
import sqlite3
from API.services.get_production_model import get_production_model
from API.services.connect_database import get_db

router = APIRouter(prefix="/health", tags=["Health Check"])

@router.get("/", description="This endpoint checks the health of the API, including database and model availability.", summary="Database and model health check")
async def health_check(conn: sqlite3.Connection = Depends(get_db)):
    
    try:
        cur = conn.cursor()
        cur.execute("SELECT * FROM algorithms LIMIT 1")
        cur.close()
        database_status = "connected"
    except Exception:
        database_status = "not connected"

    production_model = get_production_model()
    
    if production_model is None or production_model["path"].exists() is False:
        model_status = "missing"
    else:
        model_status = "available"
        
    if database_status == "connected" and model_status == "available":
        overall_status = "healthy"
    else:
        overall_status = "unhealthy"

    if overall_status == "unhealthy":
        raise HTTPException(
            status_code=503,
            detail={
                "database": database_status,
                "model": model_status
            }
        )

    return {
        "status": "healthy",
        "database": database_status,
        "model": model_status
    }
from fastapi import APIRouter, status, HTTPException, Depends
from schemas.request_model import PredictionRequest
from API.services.connect_database import get_db
import sqlite3
import numpy as np
from API.services.doe_validator import validate_doe
from API.services.feature_extractor import extract_features
from API.services.predictor import predict
from API.services.email_service import send_email
from datetime import datetime
import json

router = APIRouter(prefix="/prediction", tags=["Prediction"])

@router.post("/", response_model=str, status_code=status.HTTP_201_CREATED,  description="This endpoint predicts the best performing algorithm for a given optimisation problem based on the provided data.", summary="Predicting the best performing algorithm for a given optimisation problem.")
async def prediction(data: PredictionRequest, conn: sqlite3.Connection = Depends(get_db)):
    
    cur = conn.cursor()
    cur.execute("INSERT INTO requests (user_email, request_time, request_status, problem_name, problem_dim, num_sample_points) VALUES (?, ?, ?, ?, ?, ?)", (data.user_email, datetime.now(), 'processing', data.prob_name, data.prob_dim, data.num_sample_points))
    reuest_id = cur.lastrowid
    conn.commit()
    cur.close()
    
    try:
        validate_doe(np.array(data.doe), np.array(data.lb), np.array(data.ub), conn)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    
    features = extract_features(np.array(data.doe), np.array(data.lb), np.array(data.ub), 0, conn)
    predicted_algorithms = predict(features, conn)
    email_status = send_email(data.user_email, predicted_algorithms)
    
    cur = conn.cursor()
    cur.execute("UPDATE requests SET request_status = ? WHERE request_id = ?", ('completed', reuest_id))
    conn.commit()
    cur.close()
    
    cur = conn.cursor()
    cur.execute("INSERT INTO results (request_id, predicted_algorithms, prediction_time) VALUES (?, ?, ?)", (reuest_id, json.dumps(predicted_algorithms), datetime.now()))
    conn.commit()
    cur.close()
    
    return predicted_algorithms
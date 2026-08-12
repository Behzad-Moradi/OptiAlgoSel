from fastapi import APIRouter, status, HTTPException, Depends
import psycopg
import numpy as np
from datetime import datetime, timezone
import json
import logging
import time
from API.schemas.request_model import PredictionRequest
from API.schemas.response_model import PredictionResponse
from API.services.connect_database_pg import get_db_pg
from API.services.doe_validator import validate_doe
from API.services.feature_extractor import extract_features
from API.services.predictor import predict
from API.services.email_service import send_email

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/prediction", tags=["Prediction"])

@router.post("/", response_model=PredictionResponse, status_code=status.HTTP_201_CREATED,  description="This endpoint predicts the best performing algorithm for a given optimisation problem based on the provided data.", summary="Predicting the best performing algorithm for a given optimisation problem.")
def prediction(data: PredictionRequest, conn: psycopg.Connection = Depends(get_db_pg)):
    start_time = time.time()
    logger.info(
        "Prediction request received: problem=%s dimension=%s samples=%s",
        data.prob_name,
        data.prob_dim,
        data.num_sample_points
    )

    with conn.cursor() as cur:
        cur.execute("INSERT INTO requests (user_email, request_time, request_status, problem_name, problem_dim, num_sample_points) VALUES (%s, %s, %s, %s, %s, %s) RETURNING request_id", (data.user_email, datetime.now(timezone.utc), 'processing', data.prob_name, data.prob_dim, data.num_sample_points))
        request_id = cur.fetchone()[0]
    
        logger.info("Created prediction request record. request_id=%s", request_id)   
        
        try:
            logger.info("Validating DOE. request_id=%s", request_id)
            validate_doe(np.array(data.doe), np.array(data.lb), np.array(data.ub), conn)
            logger.info("DOE validation successful. request_id=%s", request_id)
        except ValueError as e:
            logger.info("DOE validation failed. request_id=%s", request_id)
            raise HTTPException(status_code=400, detail=str(e))
        
        logger.info("Starting feature extraction. request_id=%s", request_id)
        feature_start = time.time()
        features = extract_features(np.array(data.doe), np.array(data.lb), np.array(data.ub), 0, conn)
        feature_time = time.time() - feature_start
        logger.info("Feature extraction completed. request_id=%s duration=%.2f seconds", request_id, feature_time)
        logger.info("Running ML prediction. request_id=%s", request_id)
        predicted_algorithms = predict(features, conn)
        logger.info("Prediction completed. request_id=%s algorithms=%s", request_id, predicted_algorithms)
        logger.info("Sending prediction email. request_id=%s", request_id)
        email_status = send_email(data.user_email, predicted_algorithms)
        logger.info("Prediction email sent. request_id=%s", request_id)
        
        cur.execute("UPDATE requests SET request_status = %s WHERE request_id = %s", ('completed', request_id))
    
        
        cur.execute("INSERT INTO results (request_id, predicted_algorithms, prediction_time) VALUES (%s, %s, %s)", (request_id, json.dumps(predicted_algorithms), datetime.now(timezone.utc)))

        
        total_time = time.time() - start_time
        logger.info("Prediction request completed successfully. request_id=%s total_time=%.2f seconds", request_id, total_time)
    return PredictionResponse(request_id=request_id, predicted_algorithms=predicted_algorithms, email_sent=email_status)
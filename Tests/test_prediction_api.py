from fastapi.testclient import TestClient
from API.mainapi import app
from API.services.connect_database_pg import get_db_pg
import json
from datetime import datetime, timezone
import numpy as np

client = TestClient(app)


def test_prediction(mocker):
    
    mock_conn = mocker.MagicMock()
    mock_cursor = mock_conn.cursor.return_value
    mock_cursor.__enter__.return_value = mock_cursor
    
    request_id = 1
    mock_cursor.fetchone.return_value = (request_id,)

    def override_get_db_pg():
        return mock_conn

    app.dependency_overrides[get_db_pg] = override_get_db_pg
    
    data = {
        "user_email": "test@example.com",
        "prob_name": "BBOB",
        "prob_dim": 10,
        "num_sample_points": 2500,
        "doe": np.random.uniform(-5, 5, size=(2500, 11)).tolist(),
        "lb": (-5 * np.ones(10)).tolist(),
        "ub": (5 * np.ones(10)).tolist()
    }
    
    predicted_algorithms = ["CMAES", "XGBoost"]
    
    mock_doe_validation = mocker.patch("API.routers.prediction.validate_doe", return_value=None)
    mock_feature_extraction = mocker.patch("API.routers.prediction.extract_features", return_value=np.array([[1, 2, 3]]))
    mock_predict = mocker.patch("API.routers.prediction.predict", return_value=predicted_algorithms)
    mock_send_email = mocker.patch("API.routers.prediction.send_email", return_value="Email sent successfully.")

    response = client.post("/prediction/", json=data)
    assert response.status_code == 201
    assert response.json() == {"request_id": 1, "predicted_algorithms": predicted_algorithms, "email_sent": "Email sent successfully."}
    
    
    mock_doe_validation.assert_called_once()
    mock_feature_extraction.assert_called_once()
    mock_predict.assert_called_once()
    mock_send_email.assert_called_once_with("test@example.com", predicted_algorithms)
    
    mock_cursor.fetchone.assert_called_once_with()
    mock_cursor.execute.assert_any_call("INSERT INTO requests (user_email, request_time, request_status, problem_name, problem_dim, num_sample_points) VALUES (%s, %s, %s, %s, %s, %s) RETURNING request_id", mocker.ANY)
    mock_cursor.execute.assert_any_call("UPDATE requests SET request_status = %s WHERE request_id = %s", ("completed", request_id))
    mock_cursor.execute.assert_any_call("INSERT INTO results (request_id, predicted_algorithms, prediction_time) VALUES (%s, %s, %s)", mocker.ANY)
                                            
    app.dependency_overrides.clear()
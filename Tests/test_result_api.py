from fastapi.testclient import TestClient
from API.mainapi import app
from API.services.connect_database_pg import get_db_pg

client = TestClient(app)


def test_get_result(mocker):

    predicted_algorithms = "CMAES, Covariance Matrix Adaptation Evolution Strategy"
    request_id = 1

    mock_conn = mocker.MagicMock()
    mock_cursor = mock_conn.cursor.return_value
    mock_cursor.__enter__.return_value = mock_cursor
    
    mock_cursor.fetchone.return_value = (predicted_algorithms,)
    
    def override_get_db_pg():
        return mock_conn

    app.dependency_overrides[get_db_pg] = override_get_db_pg
    

    response = client.get(f"/result/{request_id}")
    assert response.status_code == 200
    assert response.json() == {"request_id": request_id, "predicted_algorithms": predicted_algorithms}

    mock_cursor.execute.assert_called_once_with("SELECT predicted_algorithms FROM results WHERE request_id = %s", (request_id,))
    app.dependency_overrides.clear()
    

def test_get_result_not_found(mocker):

    request_id = 999

    mock_conn = mocker.MagicMock()
    mock_cursor = mock_conn.cursor.return_value
    mock_cursor.__enter__.return_value = mock_cursor

    mock_cursor.fetchone.return_value = None

    def override_get_db_pg():
        return mock_conn

    app.dependency_overrides[get_db_pg] = override_get_db_pg

    response = client.get(f"/result/{request_id}")
    assert response.status_code == 404
    assert response.json() == {"detail": f"No result found for request ID {request_id}."}
    
    mock_cursor.execute.assert_called_once_with("SELECT predicted_algorithms FROM results WHERE request_id = %s",(request_id,))
    app.dependency_overrides.clear()
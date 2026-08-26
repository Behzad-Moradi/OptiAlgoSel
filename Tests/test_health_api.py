from fastapi.testclient import TestClient
from API.mainapi import app
from API.services.connect_database_pg import get_db_pg
import pytest

@pytest.fixture
def client():
    return TestClient(app)


def test_health_check(mocker, client):

    mock_conn = mocker.MagicMock()
    mock_cursor = mock_conn.cursor.return_value
    mock_cursor.__enter__.return_value = mock_cursor
    
    def override_get_db_pg():
        return mock_conn

    app.dependency_overrides[get_db_pg] = override_get_db_pg
    
    mock_model = mocker.patch("API.routers.health.get_production_model", return_value = {
                                                                                            "model_name": "SVM",
                                                                                            "version": "1.0.0",
                                                                                            "status": "production",
                                                                                            "file": "svm.joblib",
                                                                                            "path": mocker.MagicMock()
                                                                                            })

    mock_model.return_value["path"].exists.return_value = True

    response = client.get("/health/")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "database": "connected", "model": "available"}
    
    mock_cursor.execute.assert_called_once_with("SELECT * FROM algorithms LIMIT 1")

    app.dependency_overrides.clear()
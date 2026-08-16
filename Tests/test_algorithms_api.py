from fastapi.testclient import TestClient
from API.mainapi import app
from API.services.connect_database_pg import get_db_pg

client = TestClient(app)


def test_get_algorithms(mocker):

    algorithms = [{'algorithm_name': 'CMAES', 'algorithm_description': 'Covariance Matrix Adaptation Evolution Strategy'}, {'algorithm_name': 'ES', 'algorithm_description': 'Evolution Strategy'}, {'algorithm_name': 'GA', 'algorithm_description': 'Genetic Algorithm'}, {'algorithm_name': 'DE', 'algorithm_description': 'Differential Evolution'}, {'algorithm_name': 'PSO', 'algorithm_description': 'Particle Swarm Optimization'}, {'algorithm_name': 'NM', 'algorithm_description': 'Nelder-Mead'}, {'algorithm_name': 'PS', 'algorithm_description': 'Pattern Search'}]


    mock_conn = mocker.MagicMock()
    mock_cursor = mock_conn.cursor.return_value
    mock_cursor.__enter__.return_value = mock_cursor
    
    mock_cursor.fetchall.return_value = algorithms
    
    def override_get_db_pg():
        return mock_conn

    app.dependency_overrides[get_db_pg] = override_get_db_pg
    

    response = client.get("/algorithms/")
    assert response.status_code == 200
    assert response.json() == {"Optimisation Algorithm Portfolio": algorithms}

    mock_cursor.execute.assert_called_once_with("SELECT algorithm_name, algorithm_description FROM algorithms")

    app.dependency_overrides.clear()
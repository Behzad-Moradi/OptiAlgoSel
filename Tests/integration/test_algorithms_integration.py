from API.services.connect_database_pg import get_db_pg
from fastapi.testclient import TestClient
from API.mainapi import app



client = TestClient(app)


def test_get_algorithms_from_real_database():

    response = client.get("/algorithms/")

    assert response.status_code == 200

    data = response.json()

    assert "Optimisation Algorithm Portfolio" in data
    assert len(data["Optimisation Algorithm Portfolio"]) > 0
    

def test_database_connection():

    conn_generator = get_db_pg()
    conn = next(conn_generator)

    assert conn is not None

    conn.close()
    
    

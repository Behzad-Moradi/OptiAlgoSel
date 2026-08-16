from fastapi.testclient import TestClient
from API.mainapi import app

client = TestClient(app)


def test_welcome():

    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"Service": "OptiAlgoSel API", "Version": "1.0.0"}
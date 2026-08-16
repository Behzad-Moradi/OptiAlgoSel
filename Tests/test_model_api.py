from fastapi.testclient import TestClient
from API.mainapi import app

client = TestClient(app)


def test_get_models(mocker):

    mock_model_registry = [
                    {
                        "model_name": "Dummy",
                        "version": "1.0.0",
                        "status": "archived",
                        "file": "dummy.joblib"
                    },
                    {
                        "model_name": "Random Uniform",
                        "version": "1.0.0",
                        "status": "archived",
                        "file": "random_uniform.joblib"
                    },
                    {
                        "model_name": "SVM",
                        "version": "1.0.0",
                        "status": "production",
                        "file": "svm.joblib"
                    },
                    {
                        "model_name": "XGBoost",
                        "version": "1.0.0",
                        "status": "archived",
                        "file": "xgboost.joblib"
                    },
                    {
                        "model_name": "Random Forest",
                        "version": "1.0.0",
                        "status": "archived",
                        "file": "random_forest.joblib"
                    },
                    {
                        "model_name": "KNN",
                        "version": "1.0.0",
                        "status": "archived",
                        "file": "knn.joblib"
                    },
                    {
                        "model_name": "MLP",
                        "version": "1.0.0",
                        "status": "archived",
                        "file": "mlp.joblib"
                    }
                    ]


    mock_load_model_registry = mocker.patch("API.routers.model.load_model_registry", return_value=mock_model_registry)

    response = client.get("/models/")
    assert response.status_code == 200
    assert response.json() == {"Machine Learning Model Portfolio": mock_load_model_registry.return_value}

    mock_load_model_registry.assert_called_once_with()

    
    
from fastapi import APIRouter
from API.services.load_model_registery import load_model_registry

router = APIRouter(prefix="/models", tags=["Predictive Models"])

@router.get("/", description="This endpoint retrieves the list of predictive models available in the system.", summary="List of predictive models.")
def get_models():
    model_registry = load_model_registry()
    return {"Machine Learning Model Portfolio": model_registry}
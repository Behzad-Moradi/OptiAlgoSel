from fastapi import APIRouter

router = APIRouter(tags=["Welcome"])

@router.get("/", description="Welcome to the OptiAlgoSel API. This endpoint provides basic information about the API.", summary="Welcome to the OptiAlgoSel API.")
def welcome():
    return {"Service": "OptiAlgoSel API", "Version": "1.0.0"}
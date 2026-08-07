from pydantic import BaseModel, Field

class PredictionResponse(BaseModel):
    request_id: int = Field(..., description="The ID of the prediction request.")
    predicted_algorithms: list[str] = Field(..., description="The list of predicted optimisation algorithms.")
    email_sent: str = Field(..., description="Whether an email has been sent to the user.")
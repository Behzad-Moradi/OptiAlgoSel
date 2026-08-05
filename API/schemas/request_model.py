from pydantic import BaseModel, Field, EmailStr

class PredictionRequest(BaseModel):
    user_email: EmailStr = Field(description="The email of the user", example="user@example.com")
    prob_name: str = Field(default="RealWorld", description="The name of the optimisation problem", example="SMD1")
    prob_dim: int = Field(default=10, description="The dimension of the optimisation problem", example=10)
    num_sample_points: int = Field(default=2500, description="The number of sample points", example=2500)
    doe: list[list[float]] = Field(description="A 2D array of floats: the number of sample points BY the dimension")
    lb: list[float] = Field(description="A 1D array of floats: the lower bound of each variable")
    ub: list[float] = Field(description="A 1D array of floats: the upper bound of each variable")

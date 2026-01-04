from pydantic import BaseModel

class PredictResponse(BaseModel):
    predictions: dict[str, float]
    top_genre: str
    top_probability: float
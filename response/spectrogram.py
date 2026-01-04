from pydantic import BaseModel
from typing import List

class Spectrogram(BaseModel):
    data: List[List[List[float]]]
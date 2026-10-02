from pydantic import BaseModel
from typing import List

class PredictionRequest(BaseModel):
    alcohol: float
    malic_acid: float
    ash: float
    alcalinity_of_ash: float
    magnesium: float
    total_phenols: float
    flavanoids: float
    nonflavanoid_phenols: float
    proanthocyanins: float
    color_intensity: float
    hue: float
    od280_od315_of_diluted_wines: float
    proline: float

class BatchPredictionRequest(BaseModel):
    data: List[PredictionRequest]

class PredictionResponse(BaseModel):
    prediction: int
    model_version: str

class HealthResponse(BaseModel):
    status: str
    model_version: str

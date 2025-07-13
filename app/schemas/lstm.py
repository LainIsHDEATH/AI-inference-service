from typing import List
from pydantic import BaseModel

class LstmSensorDTO(BaseModel):
    tempIn: float
    tempOut: float
    tempSetpoint: float
    heaterPower: float

class LstmModelDtoRequest(BaseModel):
    modelId: int
    sensorDataDTOList: List[LstmSensorDTO]

class LstmModelDtoResponse(BaseModel):
    predictedTemp: float

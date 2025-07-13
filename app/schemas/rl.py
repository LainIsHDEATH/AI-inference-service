from pydantic import BaseModel, Field

class RlModelDtoRequest(BaseModel):
    modelId: int = Field(..., alias="modelId")
    tempIn: float = Field(..., alias="roomTemp")
    tempOut: float = Field(..., alias="outdoorTemp")

class RlModelDtoResponse(BaseModel):
    heaterPower: float = Field(..., alias="heaterPower")

from pydantic import BaseModel

class VerifyResponse(BaseModel):
    similarity: float

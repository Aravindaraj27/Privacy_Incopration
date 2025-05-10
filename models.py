from pydantic import BaseModel
from typing import List

class AnonymizationRequest(BaseModel):
    selected_columns: List[str]
    k: int

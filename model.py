from pydantic import BaseModel
from typing import List, Optional

class Item(BaseModel):
    item_id : Optional[int] = None
    features: List[float]
    prediction: Optional[int] = None

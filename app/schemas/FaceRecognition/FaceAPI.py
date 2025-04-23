from typing import *

from pydantic import BaseModel

class RecognizeRequest(BaseModel):
    image: str
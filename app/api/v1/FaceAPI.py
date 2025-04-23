from typing import *

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException, status, WebSocketException

from schemas.FaceRecognition.FaceAPI import RecognizeRequest
from services.FaceRecognition import predict, re_train

router = APIRouter()

@router.post("/recognize")
async def recognize(request: RecognizeRequest):
    image = request.image

    # try:
    predictions = predict(image)
    print(predictions)

    return {"predictions": predictions}

    # except Exception as err:
    #     return HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(err))




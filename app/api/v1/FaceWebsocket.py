import time
import asyncio
from typing import *

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException, status, WebSocketException

from services.FaceRecognition import re_train, predict
from services.ConnectionManager import connection_manager

router = APIRouter()

@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    try:
        await connection_manager.connect(websocket)
        id_cccd: str = None
        t0 = time.time()
        try:    
            id_cccd = await asyncio.wait_for(
            websocket.receive_text(),
            timeout = 10 - (time.time() - t0)
            )
        except asyncio.TimeoutError:
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            raise WebSocketException(code=status.WS_1008_POLICY_VIOLATION, reason = "Không nhận được ip cccd sau 10s")
            
        connection_manager.update(websocket, id_cccd)
        while True:
            data = await websocket.receive_text()
            try:
                result = predict()
                print(result)
                await connection_manager.send_response({"success": True, "event": "webcam", "payload": result}, websocket)
            except Exception as err:
                await connection_manager.send_response({"success": False, "event": "webcam", "payload": [],
                    "error": {
                        "code": status.HTTP_400_BAD_REQUEST,
                        "message": err
                    }
                }, websocket) 
    except WebSocketDisconnect:
        connection_manager.disconnect(websocket)
    except:
        connection_manager.disconnect(websocket)
        raise HTTPException(status_code = status.HTTP_500_INTERNAL_SERVER_ERROR, detail=Exception)

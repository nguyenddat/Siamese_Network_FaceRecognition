import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import uvicorn
from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

from api.v1 import FaceAPI


def get_application() -> FastAPI:
    application = FastAPI()

    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    application.include_router(FaceAPI.router, prefix="/api/v1/face")
    return application


app = get_application()
if __name__ == '__main__':
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload = True)

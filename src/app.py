"""
@author: { MUHAMMAD HASSAN KIYANI, SOFTWARE DEVELOPER, FALCONRY SOLUTIONS }
@description: This file bootstraps the application
"""

import uvicorn
from fastapi import FastAPI
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from routes import routes
import configs.configs as config
from enums.enums import ServerSettings


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Application startup: Initializing resources...")
    yield
    print("Application shutdown: Cleaning up resources...")


app = FastAPI(title=config.APP_TITLE, version=config.APP_VERSION, lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(routes.gen_ai_router)

if __name__ == "__main__":
    uvicorn.run(
        ServerSettings.SERVER_APP.value,
        host=ServerSettings.HOST.value,
        port=ServerSettings.PORT.value,
        reload=ServerSettings.SERVER_RELOAD.value,
    )

print("App file setup successful")
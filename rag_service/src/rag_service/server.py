# rag_service/server.py
from __future__ import annotations

from fastapi import FastAPI

from rag_service.settings import load_settings
from rag_service.wiring import build_app_chain
from rag_service.routes import router as api_router

app = FastAPI()

settings = load_settings("configs/rag.yaml")
app.state.chain = build_app_chain(settings)

app.include_router(api_router)

# rag_service/routes.py
from __future__ import annotations

from typing import Any, Dict
from fastapi import APIRouter, Request
from pydantic import BaseModel

router = APIRouter()

class ChatRequest(BaseModel):
    query: str

@router.post("/chat")
def chat(req: ChatRequest, request: Request) -> Dict[str, Any]:
    chain = request.app.state.chain
    return chain.invoke(req.query)

@router.get("/healthz")
def healthz():
    return {"status": "ok"}

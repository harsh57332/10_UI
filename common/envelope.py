"""Uniform request/response envelope shared by the Agent and every model service
(vLLM wrapper, cuRobo, OWLv2), so swapping one never changes how callers talk to it.

Request:  {"id": "<uuid>", "action": "<name>", "params": {...}}
Response: {"id": "<uuid>", "ok": true|false, "status": "done|running|failed|rejected",
           "result": {...}|null, "error": "<text>"|null, "ms": <float>}
"""
import time
import uuid
from typing import Any, Optional

from pydantic import BaseModel, Field


class Request(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    action: str
    params: dict = Field(default_factory=dict)


class Response(BaseModel):
    id: str
    ok: bool
    status: str = "done"
    result: Optional[Any] = None
    error: Optional[str] = None
    ms: float = 0.0


def done(req_id: str, result: Any = None, t0: Optional[float] = None) -> Response:
    return Response(id=req_id, ok=True, status="done", result=result, ms=_ms(t0))


def running(req_id: str, result: Any = None, t0: Optional[float] = None) -> Response:
    return Response(id=req_id, ok=True, status="running", result=result, ms=_ms(t0))


def failed(req_id: str, error: str, t0: Optional[float] = None, status: str = "failed") -> Response:
    return Response(id=req_id, ok=False, status=status, error=error, ms=_ms(t0))


def _ms(t0: Optional[float]) -> float:
    return round((time.time() - t0) * 1000, 1) if t0 else 0.0

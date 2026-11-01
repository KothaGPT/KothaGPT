from __future__ import annotations

import json

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from ..auth import require_api_token
from ..backend import get_active_backend
from ..schemas import Agent, AgentRun, AgentRunRequest, AgentSpec

router = APIRouter(
    prefix="/v1/agents",
    tags=["agents"],
    dependencies=[Depends(require_api_token)],
)


class AgentList(BaseModel):
    object: str = "list"
    data: list[Agent]


@router.get("", response_model=AgentList)
def list_agents() -> AgentList:
    backend = get_active_backend("agents")
    return AgentList(data=backend.list_agents())


@router.post("", response_model=Agent, status_code=201)
def create_agent(spec: AgentSpec) -> Agent:
    backend = get_active_backend("agents")
    return backend.create_agent(spec)


@router.get("/{agent_id}", response_model=Agent)
def get_agent(agent_id: str) -> Agent:
    backend = get_active_backend("agents")
    try:
        return backend.get_agent(agent_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.delete("/{agent_id}", status_code=204)
def delete_agent(agent_id: str) -> None:
    backend = get_active_backend("agents")
    try:
        backend.delete_agent(agent_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/{agent_id}/runs", response_model=AgentRun, status_code=201)
def run_agent(agent_id: str, request: AgentRunRequest) -> AgentRun:
    backend = get_active_backend("agents")
    return backend.run_agent(agent_id, request.message)


@router.post("/{agent_id}/runs/stream")
async def stream_agent(agent_id: str, request: AgentRunRequest) -> StreamingResponse:
    backend = get_active_backend("agents")
    return StreamingResponse(
        _agent_sse(backend.run_agent_stream(agent_id, request.message)),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/{agent_id}/runs/{run_id}", response_model=AgentRun)
def get_run(agent_id: str, run_id: str) -> AgentRun:
    backend = get_active_backend("agents")
    try:
        run = backend.get_run(run_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if run.agent_id != agent_id:
        raise HTTPException(status_code=404, detail="Run not found for this agent")
    return run


async def _agent_sse(stream):
    async for event in stream:
        yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
    yield "data: [DONE]\n\n"

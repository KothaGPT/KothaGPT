from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from ..auth import require_api_token
from ..backend import get_active_backend
from ..schemas import Tool, ToolInvokeRequest, ToolInvokeResponse


class ToolList(BaseModel):
    object: str = "list"
    data: list[Tool]


router = APIRouter(
    prefix="/v1/tools",
    tags=["tools"],
    dependencies=[Depends(require_api_token)],
)


@router.get("", response_model=ToolList)
def list_tools() -> ToolList:
    backend = get_active_backend("tools")
    return ToolList(data=backend.list_tools())


@router.get("/{name}", response_model=Tool)
def get_tool(name: str) -> Tool:
    backend = get_active_backend("tools")
    for tool in backend.list_tools():
        if tool.function.name == name:
            return tool
    raise HTTPException(status_code=404, detail=f"Unknown tool: {name}")


@router.post("/{name}/invoke", response_model=ToolInvokeResponse)
def invoke_tool(name: str, request: ToolInvokeRequest) -> ToolInvokeResponse:
    backend = get_active_backend("tools")
    try:
        return backend.invoke_tool(name, request.arguments)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

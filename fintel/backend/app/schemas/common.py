from __future__ import annotations
from typing import Generic, TypeVar
from pydantic import BaseModel

DataT = TypeVar("DataT")


class PaginationParams(BaseModel):
    offset: int = 0
    limit: int = 20


class PaginatedResponse(BaseModel, Generic[DataT]):
    data: list[DataT]
    total: int
    offset: int
    limit: int


class APIResponse(BaseModel, Generic[DataT]):
    data: DataT
    message: str = "success"

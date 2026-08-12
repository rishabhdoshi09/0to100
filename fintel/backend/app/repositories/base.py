from __future__ import annotations
import uuid
from typing import Any, Generic, Type, TypeVar
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from app.models.base import TimestampedBase

ModelT = TypeVar("ModelT", bound=TimestampedBase)


class BaseRepository(Generic[ModelT]):
    def __init__(self, model: Type[ModelT], session: AsyncSession) -> None:
        self._model = model
        self._session = session

    async def get_by_id(self, id: uuid.UUID) -> ModelT | None:
        return await self._session.get(self._model, id)

    async def get_by_field(self, field: str, value: Any) -> ModelT | None:
        stmt = select(self._model).where(getattr(self._model, field) == value)
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none()

    async def list(
        self, offset: int = 0, limit: int = 20, **filters: Any
    ) -> tuple[list[ModelT], int]:
        stmt = select(self._model)
        for field, value in filters.items():
            stmt = stmt.where(getattr(self._model, field) == value)
        count_stmt = select(func.count()).select_from(stmt.subquery())
        total = (await self._session.execute(count_stmt)).scalar_one()
        stmt = stmt.offset(offset).limit(limit)
        result = await self._session.execute(stmt)
        return list(result.scalars().all()), total

    async def create(self, **kwargs: Any) -> ModelT:
        obj = self._model(**kwargs)
        self._session.add(obj)
        await self._session.flush()
        await self._session.refresh(obj)
        return obj

    async def update(self, obj: ModelT, **kwargs: Any) -> ModelT:
        for key, value in kwargs.items():
            setattr(obj, key, value)
        await self._session.flush()
        await self._session.refresh(obj)
        return obj

    async def delete(self, obj: ModelT) -> None:
        await self._session.delete(obj)
        await self._session.flush()

    async def exists(self, field: str, value: Any) -> bool:
        stmt = select(func.count()).where(getattr(self._model, field) == value).select_from(self._model)
        count = (await self._session.execute(stmt)).scalar_one()
        return count > 0

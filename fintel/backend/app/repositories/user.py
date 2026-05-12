from __future__ import annotations
from sqlalchemy.ext.asyncio import AsyncSession
from app.models.user import User
from app.repositories.base import BaseRepository


class UserRepository(BaseRepository[User]):
    def __init__(self, session: AsyncSession) -> None:
        super().__init__(User, session)

    async def get_by_email(self, email: str) -> User | None:
        return await self.get_by_field("email", email)

    async def get_by_username(self, username: str) -> User | None:
        return await self.get_by_field("username", username)

from __future__ import annotations
from typing import Annotated
from fastapi import Depends, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.exceptions import AuthenticationError, AuthorizationError
from app.core.security import decode_token
from app.db.engine import get_db_session
from app.models.user import User, UserRole
from app.repositories.user import UserRepository

bearer_scheme = HTTPBearer()

DBSession = Annotated[AsyncSession, Depends(get_db_session)]


async def get_current_user(
    credentials: Annotated[HTTPAuthorizationCredentials, Depends(bearer_scheme)],
    session: DBSession,
    request: Request,
) -> User:
    try:
        payload = decode_token(credentials.credentials)
    except ValueError as exc:
        raise AuthenticationError(str(exc)) from exc

    if payload.type != "access":
        raise AuthenticationError("Invalid token type")

    redis = request.app.state.redis
    from app.services.auth import AuthService
    svc = AuthService(UserRepository(session), redis)
    if await svc.is_token_blacklisted(payload.jti):
        raise AuthenticationError("Token has been revoked")

    user = await UserRepository(session).get_by_id(__import__("uuid").UUID(payload.sub))
    if not user or not user.is_active:
        raise AuthenticationError("User not found or inactive")
    return user


CurrentUser = Annotated[User, Depends(get_current_user)]


def require_role(*roles: UserRole):
    async def _check(current_user: CurrentUser) -> User:
        if current_user.role not in roles:
            raise AuthorizationError("Insufficient permissions")
        return current_user
    return Depends(_check)


AdminUser = Annotated[User, require_role(UserRole.admin)]

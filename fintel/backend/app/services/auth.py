from __future__ import annotations
from datetime import timedelta, timezone, datetime
from app.core.config import settings
from app.core.exceptions import AuthenticationError, ConflictError
from app.core.security import (
    create_access_token,
    create_refresh_token,
    decode_token,
    hash_password,
    verify_password,
)
from app.repositories.user import UserRepository
from app.schemas.auth import LoginRequest, RegisterRequest, TokenResponse, UserOut


class AuthService:
    def __init__(self, user_repo: UserRepository, redis) -> None:
        self._users = user_repo
        self._redis = redis

    async def register(self, req: RegisterRequest) -> UserOut:
        if await self._users.exists("email", req.email):
            raise ConflictError("Email already registered")
        if await self._users.exists("username", req.username):
            raise ConflictError("Username already taken")
        user = await self._users.create(
            email=req.email,
            username=req.username,
            hashed_password=hash_password(req.password),
            full_name=req.full_name,
        )
        return UserOut.model_validate(user)

    async def login(self, req: LoginRequest) -> TokenResponse:
        user = await self._users.get_by_email(req.email)
        if not user or not verify_password(req.password, user.hashed_password):
            raise AuthenticationError("Invalid credentials")
        if not user.is_active:
            raise AuthenticationError("Account is inactive")
        return await self._issue_tokens(str(user.id))

    async def refresh(self, refresh_token: str) -> TokenResponse:
        try:
            payload = decode_token(refresh_token)
        except ValueError as exc:
            raise AuthenticationError(str(exc)) from exc

        if payload.type != "refresh":
            raise AuthenticationError("Invalid token type")

        if await self.is_token_blacklisted(payload.jti):
            raise AuthenticationError("Token has been revoked")

        # Rotate: blacklist old refresh token
        await self._blacklist_token(payload.jti, payload.exp)

        return await self._issue_tokens(payload.sub)

    async def logout(self, access_jti: str, access_exp: datetime) -> None:
        await self._blacklist_token(access_jti, access_exp)

    async def is_token_blacklisted(self, jti: str) -> bool:
        return bool(await self._redis.get(f"blacklist:{jti}"))

    async def _issue_tokens(self, user_id: str) -> TokenResponse:
        access_token, _ = create_access_token(user_id)
        refresh_token, refresh_jti = create_refresh_token(user_id)

        # Store refresh token in Redis for validation
        ttl = settings.refresh_token_expire_days * 86400
        await self._redis.setex(f"refresh_token:{refresh_jti}", ttl, user_id)

        return TokenResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_in=settings.access_token_expire_minutes * 60,
        )

    async def _blacklist_token(self, jti: str, exp: datetime) -> None:
        now = datetime.now(timezone.utc)
        ttl = max(int((exp - now).total_seconds()), 1)
        await self._redis.setex(f"blacklist:{jti}", ttl, "1")

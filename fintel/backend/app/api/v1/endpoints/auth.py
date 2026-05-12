from __future__ import annotations
from fastapi import APIRouter, Request
from app.api.deps import CurrentUser, DBSession
from app.db.redis import get_redis
from app.repositories.user import UserRepository
from app.schemas.auth import LoginRequest, RefreshRequest, RegisterRequest, TokenResponse, UserOut
from app.services.auth import AuthService

router = APIRouter(prefix="/auth", tags=["auth"])


def _get_service(session, redis) -> AuthService:
    return AuthService(UserRepository(session), redis)


@router.post("/register", response_model=UserOut, status_code=201)
async def register(req: RegisterRequest, session: DBSession, request: Request):
    redis = request.app.state.redis
    return await _get_service(session, redis).register(req)


@router.post("/login", response_model=TokenResponse)
async def login(req: LoginRequest, session: DBSession, request: Request):
    redis = request.app.state.redis
    return await _get_service(session, redis).login(req)


@router.post("/refresh", response_model=TokenResponse)
async def refresh(req: RefreshRequest, session: DBSession, request: Request):
    redis = request.app.state.redis
    return await _get_service(session, redis).refresh(req.refresh_token)


@router.get("/me", response_model=UserOut)
async def me(current_user: CurrentUser):
    return UserOut.model_validate(current_user)


@router.post("/logout", status_code=204)
async def logout(current_user: CurrentUser, request: Request, session: DBSession):
    from app.core.security import decode_token
    auth_header = request.headers.get("Authorization", "")
    token = auth_header.replace("Bearer ", "")
    payload = decode_token(token)
    redis = request.app.state.redis
    await _get_service(session, redis).logout(payload.jti, payload.exp)

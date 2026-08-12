from __future__ import annotations
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse


class FintelError(Exception):
    status_code: int = 500
    detail: str = "Internal server error"

    def __init__(self, detail: str | None = None):
        self.detail = detail or self.__class__.detail
        super().__init__(self.detail)


class NotFoundError(FintelError):
    status_code = 404
    detail = "Resource not found"


class AuthenticationError(FintelError):
    status_code = 401
    detail = "Authentication required"


class AuthorizationError(FintelError):
    status_code = 403
    detail = "Insufficient permissions"


class ConflictError(FintelError):
    status_code = 409
    detail = "Resource already exists"


class ValidationError(FintelError):
    status_code = 422
    detail = "Validation error"


class RateLimitError(FintelError):
    status_code = 429
    detail = "Rate limit exceeded"


class ExternalServiceError(FintelError):
    status_code = 502
    detail = "External service error"


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(FintelError)
    async def fintel_error_handler(request: Request, exc: FintelError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": exc.detail, "type": type(exc).__name__},
        )

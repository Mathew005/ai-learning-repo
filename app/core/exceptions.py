from fastapi import Request, status
from fastapi.responses import JSONResponse
from typing import Any, Dict

class AIModelError(Exception):
    """Base exception for AI model related errors."""
    def __init__(self, message: str, details: Dict[str, Any] = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)

async def ai_model_exception_handler(request: Request, exc: AIModelError):
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={
            "error": "AI Model Service Error",
            "message": exc.message,
            "details": exc.details
        }
    )

async def generic_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal Server Error",
            "message": "An unexpected error occurred."
        }
    )

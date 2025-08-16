from functools import wraps
import logging
from typing import Callable, Any

from fastapi import HTTPException, status

from service.core.exceptions import ServiceError

logger = logging.getLogger(__name__)


def common_exception_handler(func: Callable) -> Callable:
    """
    Simplified common exception handler for FastAPI routes.
    
    Maps common exceptions to appropriate HTTP status codes:
    - ServiceError: Uses the exception's http_status
    - ValueError: 400 Bad Request
    - Exception: 500 Internal Server Error
    """
    @wraps(func)
    async def inner_function(*args: Any, **kwargs: Any) -> Any:
        try:
            return await func(*args, **kwargs)
        except ServiceError as e:
            raise HTTPException(e.http_status, str(e)) from e
        except ValueError as e:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, f"Invalid input: {e}") from e
        except FileNotFoundError as e:
            raise HTTPException(status.HTTP_404_NOT_FOUND, f"Resource not found: {e}") from e
        except HTTPException:
            raise
        except Exception as e:
            logger.exception("Unexpected error in %s", func.__name__)
            raise HTTPException(
                status.HTTP_500_INTERNAL_SERVER_ERROR, 
                "Internal server error"
            ) from e
    
    return inner_function

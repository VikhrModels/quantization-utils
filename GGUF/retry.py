import logging
import time
from functools import wraps
from typing import Callable, Tuple, Type

from exceptions import NetworkError

logger = logging.getLogger(__name__)


def retry_on_network_error(
    max_retries: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
):
    """
    Decorator to retry function on network errors with exponential backoff

    Args:
            max_retries: Maximum number of retry attempts
            delay: Initial delay between retries in seconds
            backoff: Backoff multiplier (e.g., 2.0 means double the delay each time)
            exceptions: Tuple of exception types to catch and retry
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            current_delay = delay
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e

                    if attempt < max_retries:
                        logger.warning(
                            f"Attempt {attempt + 1}/{max_retries + 1} failed for {func.__name__}: {e}. "
                            f"Retrying in {current_delay:.1f}s..."
                        )
                        time.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        logger.error(
                            f"All {max_retries + 1} attempts failed for {func.__name__}"
                        )

            operation_name = f"{func.__name__}"
            if last_exception:
                raise NetworkError(
                    operation=operation_name,
                    reason=str(last_exception),
                    retry_count=max_retries,
                ) from last_exception
            else:
                raise NetworkError(
                    operation=operation_name,
                    reason="Unknown error",
                    retry_count=max_retries,
                )

        return wrapper

    return decorator

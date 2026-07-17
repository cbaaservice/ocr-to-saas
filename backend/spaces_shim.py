from __future__ import annotations

from collections.abc import Callable
from typing import Any


try:
    import spaces  # Must be imported before torch on Hugging Face ZeroGPU.
except ImportError:  # Local development uses a no-op decorator.

    class _LocalSpaces:
        @staticmethod
        def GPU(*decorator_args: Any, **decorator_kwargs: Any) -> Callable:
            def decorate(function: Callable) -> Callable:
                return function

            if decorator_args and callable(decorator_args[0]) and len(decorator_args) == 1:
                return decorator_args[0]
            return decorate

    spaces = _LocalSpaces()  # type: ignore[assignment]

__all__ = ["spaces"]

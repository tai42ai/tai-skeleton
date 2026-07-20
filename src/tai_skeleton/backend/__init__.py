"""Backend feature: the execution-backend seam + callback glue.

The :class:`~tai_contract.backend.Backend` ABC is the contract; concrete worker
backends (celery/rq/arq) are external plugins that implement it and register via
``@tai_app.backends.register_backend``. This package re-exports the ABC as the registration
seam and ships the callback glue (the :class:`CallbackSchema` impl plus
``callback_execution`` / ``prepare_backend_kwargs``).
"""

from tai_contract.backend import Backend

from tai_skeleton.backend.callback import (
    CallbackSchema,
    callback_execution,
    prepare_backend_kwargs,
)

__all__ = [
    "Backend",
    "CallbackSchema",
    "callback_execution",
    "prepare_backend_kwargs",
]

"""Structured logging configuration.

Sets up structlog to write JSON Lines to stdout AND to an in-memory
deque so the Streamlit dashboard can read recent events without touching
the filesystem.
"""

from __future__ import annotations

import datetime
import json
import logging
from collections import deque

import structlog

_EVENT_BUFFER: deque[dict] = deque(maxlen=200)


class _DequeProcessor:
    """structlog processor that appends each event to the in-memory deque."""

    def __call__(self, logger: object, method: str, event_dict: dict) -> dict:  # noqa: ARG002
        _EVENT_BUFFER.append(dict(event_dict))
        return event_dict


def configure_logging() -> None:
    """Call once at startup. Idempotent."""
    shared_processors = [
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        _DequeProcessor(),
    ]

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.DEBUG),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_recent_events(n: int = 15) -> list[dict]:
    """Return up to *n* most-recent log events as plain dicts."""
    events = list(_EVENT_BUFFER)
    return events[-n:]

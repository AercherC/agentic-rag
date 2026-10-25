"""Minimal thread-isolation helper for components shared across concurrent requests."""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

_STATE_ATTRIBUTE = "_evidenceflow_per_thread_state"


class PerThread:
    """Instance-attribute descriptor that gives every thread its own value, without clobbering.

    The runtime shares the Generator, LLM client and reranker as process-wide singletons; their
    ``last_*`` diagnostics only describe "the most recent call on this thread". A plain instance
    attribute would let concurrent requests overwrite each other's values. With this descriptor the
    existing ``self.x = ...`` code stays unchanged, but each thread only sees its own value. On the
    first read of a thread that never wrote, ``default_factory()`` produces the initial value.
    """

    def __init__(self, default_factory: Callable[[], Any]) -> None:
        self._default_factory = default_factory
        self._name = ""

    def __set_name__(self, owner: type, name: str) -> None:
        self._name = name

    def _local(self, instance: Any) -> threading.local:
        # dict.setdefault is atomic in CPython, so two threads touching this attribute for the first
        # time cannot each create their own threading.local().
        return instance.__dict__.setdefault(_STATE_ATTRIBUTE, threading.local())

    def __get__(self, instance: Any, owner: type | None = None) -> Any:
        if instance is None:
            return self
        local = self._local(instance)
        try:
            return getattr(local, self._name)
        except AttributeError:
            value = self._default_factory()
            setattr(local, self._name, value)
            return value

    def __set__(self, instance: Any, value: Any) -> None:
        setattr(self._local(instance), self._name, value)

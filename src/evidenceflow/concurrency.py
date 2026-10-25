"""共享组件在并发请求下使用的最小线程隔离工具。"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

_STATE_ATTRIBUTE = "_evidenceflow_per_thread_state"


class PerThread:
    """实例属性描述符：每个线程各自读写一份值，互不覆盖。

    Runtime 把 Generator、LLM Client、Reranker 作为进程内单例共享；它们的 ``last_*``
    诊断只表示“本线程最近一次调用”。普通实例属性在并发请求下会被别的线程覆盖，
    换成本描述符后原有 ``self.x = ...`` 写法不变，但每个线程只看到自己的值。
    线程第一次读取且尚未写入时，用 ``default_factory()`` 生成初值。
    """

    def __init__(self, default_factory: Callable[[], Any]) -> None:
        self._default_factory = default_factory
        self._name = ""

    def __set_name__(self, owner: type, name: str) -> None:
        self._name = name

    def _local(self, instance: Any) -> threading.local:
        # dict.setdefault 在 CPython 中是原子的，两个线程首次访问时不会各建一份 local。
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

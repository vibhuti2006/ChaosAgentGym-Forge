"""Tiny in-process mock API for the user-record domain.

The API exposes three operations: GET, PUT, VERIFY. Internally it keeps two
parallel stores:
  - `visible`: what GET sees (subject to stale reads).
  - `truth`:   what VERIFY is judged against (subject to partial writes).

Under no-failure conditions the two are identical. The chaos injector is what
makes them diverge.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

from .chaos_injector import ChaosInjector, Failure


@dataclass
class ApiResponse:
    status: int
    body: dict | None = None
    failure: Failure = Failure.NONE

    def to_text(self) -> str:
        if self.status == 503:
            return "HTTP 503 Service Unavailable"
        if self.status == 404:
            return "HTTP 404 Not Found"
        if self.body is None:
            return f"HTTP {self.status}"
        return f"HTTP {self.status} {self.body}"


@dataclass
class MockUserApi:
    injector: ChaosInjector
    truth: dict[str, dict[str, Any]] = field(default_factory=dict)
    visible: dict[str, dict[str, Any]] = field(default_factory=dict)
    # snapshot of the previous visible state, used to serve stale reads.
    _prev_visible: dict[str, dict[str, Any]] = field(default_factory=dict)
    call_count: int = 0

    @classmethod
    def with_user(cls, injector: ChaosInjector, user_id: str,
                  record: dict[str, Any]) -> "MockUserApi":
        api = cls(injector=injector)
        api.truth[user_id] = copy.deepcopy(record)
        api.visible[user_id] = copy.deepcopy(record)
        api._prev_visible[user_id] = copy.deepcopy(record)
        return api

    # --- operations -------------------------------------------------------

    def get(self, user_id: str) -> ApiResponse:
        self.call_count += 1
        fail = self.injector.roll_get()
        if fail is Failure.SERVICE_UNAVAILABLE:
            return ApiResponse(503, failure=fail)
        if user_id not in self.visible:
            return ApiResponse(404, failure=Failure.NONE)
        if fail is Failure.STALE_READ and user_id in self._prev_visible:
            return ApiResponse(200, body=copy.deepcopy(self._prev_visible[user_id]),
                               failure=fail)
        return ApiResponse(200, body=copy.deepcopy(self.visible[user_id]),
                           failure=Failure.NONE)

    def put(self, user_id: str, patch: dict[str, Any]) -> ApiResponse:
        self.call_count += 1
        fail = self.injector.roll_put()
        if fail is Failure.SERVICE_UNAVAILABLE:
            return ApiResponse(503, failure=fail)
        if user_id not in self.visible:
            return ApiResponse(404, failure=Failure.NONE)

        # snapshot before mutating so stale reads have something to serve.
        self._prev_visible[user_id] = copy.deepcopy(self.visible[user_id])
        self.visible[user_id].update(patch)

        if fail is Failure.PARTIAL_WRITE:
            # truth does NOT receive the update; agent will be misled by GET.
            return ApiResponse(200, body=copy.deepcopy(self.visible[user_id]),
                               failure=fail)

        self.truth[user_id].update(patch)
        return ApiResponse(200, body=copy.deepcopy(self.visible[user_id]),
                           failure=Failure.NONE)

    def verify_truth(self, user_id: str, expected: dict[str, Any]) -> bool:
        """NOT exposed to the agent — used by the env to score VERIFY actions."""
        record = self.truth.get(user_id, {})
        return all(record.get(k) == v for k, v in expected.items())

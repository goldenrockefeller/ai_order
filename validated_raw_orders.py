"""Fetch and represent validated raw orders from the customer API.

This module intentionally has no model, LangGraph, database, or query code.
The caller must pass the returned orders to later stages rather than
fetching the customer API again.
"""

from __future__ import annotations

import json
import logging
import socket
import time
from dataclasses import dataclass
from json import JSONDecodeError
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DEFAULT_API_URL = "http://localhost:5001/api/orders"
DEFAULT_TIMEOUT_SECONDS = 10.0

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ValidatedRawOrders:
    """The validated raw orders for one run.

    A tuple is used instead of the API's list so later code cannot mutate the
    run's source data accidentally.
    """

    raw_orders: tuple[str, ...]

    def __len__(self) -> int:
        return len(self.raw_orders)


def validate_orders_response(payload: Any) -> ValidatedRawOrders:
    """Validate an API payload and return its immutable raw orders."""

    if not isinstance(payload, dict):
        raise ValueError("API response root must be a JSON object")

    if payload.get("status") != "ok":
        raise ValueError(
            "API response field 'status' must equal 'ok'"
        )

    if "raw_orders" not in payload:
        raise ValueError("API response is missing 'raw_orders'")

    raw_orders = payload["raw_orders"]
    if not isinstance(raw_orders, list):
        raise ValueError("API response field 'raw_orders' must be a list")

    for index, order in enumerate(raw_orders):
        if not isinstance(order, str) or not order.strip():
            raise ValueError(
                "API response field 'raw_orders' must contain only non-empty strings; "
                f"item {index} is invalid"
            )

    return ValidatedRawOrders(raw_orders=tuple(raw_orders))


def _is_timeout(error: BaseException) -> bool:
    reason = getattr(error, "reason", None)
    return isinstance(error, (TimeoutError, socket.timeout)) or isinstance(
        reason, (TimeoutError, socket.timeout)
    )


def _error_text(error: BaseException) -> str:
    return str(error) or error.__class__.__name__


def fetch_validated_raw_orders(
    api_url: str = DEFAULT_API_URL,
    *,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    opener: Callable[..., Any] | None = None,
) -> ValidatedRawOrders:
    """Fetch ``/api/orders`` once and return validated, immutable raw orders.

    ``opener`` is injectable only to make transport behavior testable. The
    production default is :func:`urllib.request.urlopen`, and no query
    parameters are added to the URL.
    """

    request = Request(api_url, method="GET")
    open_url = opener or urlopen
    started = time.perf_counter()
    received_count: int | None = None

    try:
        with open_url(request, timeout=timeout) as response:
            status_code = getattr(response, "status", None)
            if status_code is None:
                status_code = response.getcode()
            if not 200 <= status_code < 300:
                raise ConnectionError(
                    f"Customer API returned HTTP {status_code} for {api_url}"
                )

            try:
                body = response.read()
            except Exception as error:
                if _is_timeout(error):
                    raise TimeoutError(
                        f"Timed out reading customer API response from {api_url}"
                    ) from error
                raise ConnectionError(
                    f"Could not read customer API response from {api_url}: "
                    f"{_error_text(error)}"
                ) from error
    except (ConnectionError, TimeoutError) as error:
        duration = time.perf_counter() - started
        logger.error(
            "Orders API request failed: url=%s outcome=%s duration_ms=%.1f error=%s",
            api_url,
            _outcome_for(error),
            duration * 1000,
            error,
        )
        raise
    except HTTPError as error:
        duration = time.perf_counter() - started
        failure = ConnectionError(
            f"Customer API returned HTTP {error.code} for {api_url}"
        )
        logger.error(
            "Orders API request failed: url=%s outcome=http_error duration_ms=%.1f error=%s",
            api_url,
            duration * 1000,
            failure,
        )
        raise failure from error
    except URLError as error:
        duration = time.perf_counter() - started
        outcome = "timeout" if _is_timeout(error) else "transport_error"
        message = (
            f"Timed out connecting to customer API at {api_url}"
            if outcome == "timeout"
            else f"Could not connect to customer API at {api_url}: {_error_text(error)}"
        )
        failure = (
            TimeoutError(message)
            if outcome == "timeout"
            else ConnectionError(message)
        )
        logger.error(
            "Orders API request failed: url=%s outcome=%s duration_ms=%.1f error=%s",
            api_url,
            outcome,
            duration * 1000,
            failure,
        )
        raise failure from error
    except OSError as error:
        duration = time.perf_counter() - started
        failure = ConnectionError(
            f"Could not connect to customer API at {api_url}: {_error_text(error)}"
        )
        logger.error(
            "Orders API request failed: url=%s outcome=transport_error duration_ms=%.1f error=%s",
            api_url,
            duration * 1000,
            failure,
        )
        raise failure from error

    try:
        payload = json.loads(body)
    except (JSONDecodeError, UnicodeDecodeError, TypeError) as error:
        duration = time.perf_counter() - started
        failure = ValueError(
            f"Customer API returned invalid JSON from {api_url}: {_error_text(error)}"
        )
        logger.error(
            "Orders API request failed: url=%s outcome=invalid_json duration_ms=%.1f error=%s",
            api_url,
            duration * 1000,
            failure,
        )
        raise failure from error

    try:
        validated_orders = validate_orders_response(payload)
        received_count = len(validated_orders)
    except ValueError as error:
        duration = time.perf_counter() - started
        if isinstance(payload, dict) and isinstance(payload.get("raw_orders"), list):
            received_count = len(payload["raw_orders"])
        logger.error(
            "Orders API request failed: url=%s outcome=validation_failed duration_ms=%.1f "
            "received_count=%s error=%s",
            api_url,
            duration * 1000,
            received_count if received_count is not None else "unknown",
            error,
        )
        raise

    duration = time.perf_counter() - started
    logger.info(
        "Orders API request complete: url=%s outcome=validated duration_ms=%.1f "
        "received_count=%d",
        api_url,
        duration * 1000,
        received_count,
    )
    return validated_orders


def _outcome_for(error: Exception) -> str:
    if isinstance(error, TimeoutError):
        return "timeout"
    if isinstance(error, ConnectionError) and "HTTP " in str(error):
        return "http_error"
    return "transport_error"

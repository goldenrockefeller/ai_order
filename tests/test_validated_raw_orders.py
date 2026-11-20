from __future__ import annotations

import json
import socket
from unittest import TestCase
from unittest.mock import Mock
from urllib.error import URLError

from fixtures.raw_orders import RAW_ORDERS as ORDERS
from validated_raw_orders import (
    DEFAULT_API_URL,
    fetch_validated_raw_orders,
    validate_orders_response,
)


class _Response:
    def __init__(
        self,
        payload: object | None = None,
        *,
        status: int = 200,
        raw_body: bytes | None = None,
    ):
        self.status = status
        self._body = raw_body if raw_body is not None else json.dumps(payload).encode()

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return self._body


class ValidatedRawOrdersCorrectnessTests(TestCase):
    def test_fetches_all_five_orders_once_without_limit(self):
        opener = Mock(
            return_value=_Response({"status": "ok", "raw_orders": ORDERS})
        )

        validated_orders = fetch_validated_raw_orders(DEFAULT_API_URL, opener=opener)

        self.assertEqual(validated_orders.raw_orders, tuple(ORDERS))
        opener.assert_called_once()
        request = opener.call_args.args[0]
        self.assertEqual(request.full_url, DEFAULT_API_URL)
        self.assertNotIn("limit", request.full_url)

    def test_missing_raw_orders_fails(self):
        opener = Mock(return_value=_Response({"status": "ok"}))
        with self.assertRaisesRegex(ValueError, "missing 'raw_orders'"):
            fetch_validated_raw_orders(DEFAULT_API_URL, opener=opener)

    def test_non_list_raw_orders_fails(self):
        with self.assertRaisesRegex(ValueError, "must be a list"):
            validate_orders_response({"status": "ok", "raw_orders": "order"})

    def test_non_string_order_fails(self):
        opener = Mock(
            return_value=_Response(
                {"status": "ok", "raw_orders": ["ok", 3]}
            )
        )
        with self.assertRaisesRegex(ValueError, "item 1 is invalid"):
            fetch_validated_raw_orders(DEFAULT_API_URL, opener=opener)

    def test_invalid_json_fails_explicitly(self):
        opener = Mock(return_value=_Response(raw_body=b"not json"))

        with self.assertRaisesRegex(ValueError, "invalid JSON"):
            fetch_validated_raw_orders(DEFAULT_API_URL, opener=opener)

    def test_http_failure_fails_explicitly(self):
        opener = Mock(return_value=_Response({}, status=503))

        with self.assertRaisesRegex(ConnectionError, "HTTP 503"):
            fetch_validated_raw_orders(DEFAULT_API_URL, opener=opener)

    def test_timeout_fails_explicitly(self):
        def timeout_opener(*_args, **_kwargs):
            raise URLError(socket.timeout("timed out"))

        with self.assertRaises(TimeoutError):
            fetch_validated_raw_orders(DEFAULT_API_URL, opener=timeout_opener)


if __name__ == "__main__":
    import unittest

    unittest.main()

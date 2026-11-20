from __future__ import annotations

from unittest import TestCase

from database import Order
from main import result_json, run_pipeline
from query_agent import FilterCondition, QueryFilter
from sample_orders import ORDERS
from validated_raw_orders import ValidatedRawOrders


EXPECTED_ORDERS = {
    raw_order: order
    for raw_order, order in zip(
        ORDERS,
        (
            Order("1001", "John Davis", "OH", 74210),
            Order("1002", "Sarah Liu", "TX", 15655),
            Order("1003", "Mike Turner", "OH", 129999),
            Order("1004", "Rachel Kim", "WA", 8950),
            Order("1005", "Chris Myers", "OH", 51200),
        ),
        strict=True,
    )
}


class _OrderAgent:
    def __init__(self, failing_order: str | None = None):
        self.failing_order = failing_order
        self.calls: list[str] = []

    def extract(self, raw_order: str) -> Order:
        self.calls.append(raw_order)
        if raw_order == self.failing_order:
            raise ValueError("forced extraction failure")
        return EXPECTED_ORDERS[raw_order]


class _QueryAgent:
    def __init__(self, filters: dict[str, QueryFilter]):
        self.filters = filters
        self.calls: list[str] = []

    def interpret(self, request: str) -> QueryFilter:
        self.calls.append(request)
        return self.filters[request]


SAMPLE_REQUEST = "Ohio orders over $500"
SECOND_REQUEST = "Orders for Sarah or shipped to Washington"
FILTERS = {
    SAMPLE_REQUEST: QueryFilter(
        "all",
        (
            FilterCondition("state", "eq", ("OH",)),
            FilterCondition("total", "gt", ("$500",)),
        ),
    ),
    SECOND_REQUEST: QueryFilter(
        "any",
        (
            FilterCondition("buyer", "contains", ("Sarah",)),
            FilterCondition("state", "in", ("WA",)),
        ),
    ),
}


class PipelineCorrectnessTests(TestCase):
    def test_two_requests_reuse_one_fetched_and_extracted_snapshot(self):
        fetch_count = 0

        def fetch_orders():
            nonlocal fetch_count
            fetch_count += 1
            return ValidatedRawOrders(tuple(ORDERS))

        order_agent = _OrderAgent()
        query_agent = _QueryAgent(FILTERS)

        results = run_pipeline(
            [SAMPLE_REQUEST, SECOND_REQUEST],
            order_agent=order_agent,
            query_agent=query_agent,
            fetch_orders=fetch_orders,
        )

        self.assertEqual(fetch_count, 1)
        self.assertEqual(order_agent.calls, ORDERS)
        self.assertEqual(query_agent.calls, [SAMPLE_REQUEST, SECOND_REQUEST])
        self.assertEqual(
            [order.order_id for order in results[0].orders],
            ["1001", "1003", "1005"],
        )
        self.assertEqual(
            [order.order_id for order in results[1].orders],
            ["1002", "1004"],
        )
        self.assertFalse(results[0].incomplete)
        self.assertFalse(results[1].incomplete)

    def test_extraction_failure_continues_and_marks_output_incomplete(self):
        result = run_pipeline(
            [SAMPLE_REQUEST],
            order_agent=_OrderAgent(failing_order=ORDERS[2]),
            query_agent=_QueryAgent(FILTERS),
            fetch_orders=lambda: ValidatedRawOrders(tuple(ORDERS)),
        )[0]

        self.assertEqual(
            [order.order_id for order in result.orders], ["1001", "1005"]
        )
        self.assertEqual(
            result_json(result),
            {
                "orders": [
                    {
                        "orderId": "1001",
                        "buyer": "John Davis",
                        "state": "OH",
                        "total": 742.1,
                    },
                    {
                        "orderId": "1005",
                        "buyer": "Chris Myers",
                        "state": "OH",
                        "total": 512.0,
                    },
                ],
                "incomplete": True,
                "failedExtractions": 1,
            },
        )


if __name__ == "__main__":
    import unittest

    unittest.main()

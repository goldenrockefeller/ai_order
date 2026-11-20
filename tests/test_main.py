from __future__ import annotations

from unittest import TestCase
from unittest.mock import patch

from database import Order
from main import result_json, run_pipeline
from order_extraction import OrderStructureAgent
from query_agent import FilterCondition, QueryAgent, QueryFilter
from fixtures.raw_orders import RAW_ORDERS as ORDERS
from tests.fixtures.labeled_orders import LABELED_ORDERS
from validated_raw_orders import ValidatedRawOrders
from voting import NoQuorumError


EXPECTED_ORDERS = {
    raw_order: order
    for raw_order, order in zip(
        ORDERS,
        LABELED_ORDERS,
        strict=True,
    )
}

LABELED_OUTPUTS = {
    ORDERS[0]: {"order_id": "1001", "buyer": "John Davis", "state": "OH", "total": "$742.10"},
    ORDERS[1]: {"order_id": "1002", "buyer": "Sarah Liu", "state": "TX", "total": "$156.55"},
    ORDERS[2]: {"order_id": "1003", "buyer": "Mike Turner", "state": "OH", "total": "$1299.99"},
    ORDERS[3]: {"order_id": "1004", "buyer": "Rachel Kim", "state": "WA", "total": "$89.50"},
    ORDERS[4]: {"order_id": "1005", "buyer": "Chris Myers", "state": "OH", "total": "$512.00"},
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


class _PerOrderVotingModel:
    def __init__(self, no_quorum_order: str | None = None):
        self.no_quorum_order = no_quorum_order
        self.counts = {raw_order: 0 for raw_order in ORDERS}

    def invoke(self, messages):
        content = messages[-1]["content"]
        raw_order = next(raw for raw in ORDERS if raw in content)
        vote_index = self.counts[raw_order]
        self.counts[raw_order] += 1
        output = LABELED_OUTPUTS[raw_order].copy()
        if raw_order == self.no_quorum_order:
            buyer_parts = output["buyer"].split()
            output["buyer"] = (output["buyer"], *buyer_parts)[vote_index]
        elif vote_index == 2:
            output["buyer"] = output["buyer"].split()[0]
        return output


class _SequenceModel:
    def __init__(self, outputs):
        self.outputs = iter(outputs)

    def invoke(self, _messages):
        return next(self.outputs)


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

    def test_no_quorum_order_is_skipped_and_marks_every_result_incomplete(self):
        results = run_pipeline(
            [SAMPLE_REQUEST, SECOND_REQUEST],
            order_agent=OrderStructureAgent(
                _PerOrderVotingModel(no_quorum_order=ORDERS[2]),
                attempts=3,
                quorum=2,
            ),
            query_agent=_QueryAgent(FILTERS),
            fetch_orders=lambda: ValidatedRawOrders(tuple(ORDERS)),
        )

        self.assertEqual(
            [order.order_id for order in results[0].orders], ["1001", "1005"]
        )
        self.assertTrue(all(result.incomplete for result in results))
        self.assertTrue(all(result.failed_extractions == 1 for result in results))

    def test_quorum_success_preserves_sample_result(self):
        query_a = {
            "logic": "all",
            "conditions": [
                {"field": "state", "operator": "eq", "values": ["OH"]},
                {"field": "total", "operator": "gt", "values": ["$500"]},
            ],
            "unsupported_conditions": [],
        }
        query_b = {
            **query_a,
            "conditions": [
                {"field": "state", "operator": "eq", "values": ["OH"]},
                {"field": "total", "operator": "gte", "values": ["$500"]},
            ],
        }
        result = run_pipeline(
            [SAMPLE_REQUEST],
            order_agent=OrderStructureAgent(
                _PerOrderVotingModel(), attempts=3, quorum=2
            ),
            query_agent=QueryAgent(
                _SequenceModel((query_a, query_a, query_b)),
                attempts=3,
                quorum=2,
            ),
            fetch_orders=lambda: ValidatedRawOrders(tuple(ORDERS)),
        )[0]

        self.assertEqual(
            [order.order_id for order in result.orders],
            ["1001", "1003", "1005"],
        )
        self.assertFalse(result.incomplete)

    def test_no_quorum_query_never_reaches_sqlite_query(self):
        query_outputs = tuple(
            {
                "logic": "all",
                "conditions": [
                    {"field": "state", "operator": "eq", "values": [state]}
                ],
                "unsupported_conditions": [],
            }
            for state in ("OH", "TX", "WA")
        )
        query_agent = QueryAgent(
            _SequenceModel(query_outputs), attempts=3, quorum=2
        )

        with patch("main.query_by_filter") as sqlite_query:
            with self.assertRaises(NoQuorumError):
                run_pipeline(
                    [SAMPLE_REQUEST],
                    order_agent=_OrderAgent(),
                    query_agent=query_agent,
                    fetch_orders=lambda: ValidatedRawOrders(tuple(ORDERS)),
                )

        sqlite_query.assert_not_called()


if __name__ == "__main__":
    import unittest

    unittest.main()

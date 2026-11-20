from __future__ import annotations

import os
from unittest import TestCase
from unittest.mock import patch

from database import Order
from order_extraction import (
    OrderStructureAgent,
    create_openrouter_order_agent,
)
from sample_orders import ORDERS


LABELED_OUTPUTS = {
    ORDERS[0]: {
        "order_id": "1001",
        "buyer": "John Davis",
        "state": "OH",
        "total": "$742.10",
    },
    ORDERS[1]: {
        "order_id": "1002",
        "buyer": "Sarah Liu",
        "state": "TX",
        "total": "$156.55",
    },
    ORDERS[2]: {
        "order_id": "1003",
        "buyer": "Mike Turner",
        "state": "OH",
        "total": "$1299.99",
    },
    ORDERS[3]: {
        "order_id": "1004",
        "buyer": "Rachel Kim",
        "state": "WA",
        "total": "$89.50",
    },
    ORDERS[4]: {
        "order_id": "1005",
        "buyer": "Chris Myers",
        "state": "OH",
        "total": "$512.00",
    },
}

EXPECTED_ORDERS = [
    Order("1001", "John Davis", "OH", 74210),
    Order("1002", "Sarah Liu", "TX", 15655),
    Order("1003", "Mike Turner", "OH", 129999),
    Order("1004", "Rachel Kim", "WA", 8950),
    Order("1005", "Chris Myers", "OH", 51200),
]


class _LabeledModel:
    def __init__(self):
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages)
        user_message = messages[-1]["content"]
        for raw_order, output in LABELED_OUTPUTS.items():
            if raw_order in user_message:
                return output.copy()
        raise AssertionError("test model received an unknown order")


class _FixedModel:
    def __init__(self, output):
        self.output = output
        self.call_count = 0

    def invoke(self, _messages):
        self.call_count += 1
        return self.output


class _FailingModel:
    def invoke(self, _messages):
        raise RuntimeError("provider unavailable")


class OrderStructureCorrectnessTests(TestCase):
    def test_all_five_orders_match_hand_labeled_records_field_for_field(self):
        model = _LabeledModel()
        agent = OrderStructureAgent(model)

        actual = [agent.extract(raw_order) for raw_order in ORDERS]

        self.assertEqual(actual, EXPECTED_ORDERS)
        self.assertEqual(len(model.calls), 5)

    def test_missing_total_is_rejected_after_model_returns_null(self):
        output = LABELED_OUTPUTS[ORDERS[0]].copy()
        output["total"] = None
        model = _FixedModel(output)

        with self.assertRaisesRegex(ValueError, "'total' is missing"):
            OrderStructureAgent(model).extract(ORDERS[0])

        self.assertEqual(model.call_count, 1)

    def test_invalid_raw_input_is_rejected_before_model_call(self):
        model = _FixedModel(LABELED_OUTPUTS[ORDERS[0]])

        for raw_order, expected_error in (("", ValueError), ("   ", ValueError), (None, TypeError)):
            with self.subTest(raw_order=raw_order):
                with self.assertRaises(expected_error):
                    OrderStructureAgent(model).extract(raw_order)

        self.assertEqual(model.call_count, 0)

    def test_model_failure_is_reported_explicitly(self):
        with self.assertRaisesRegex(RuntimeError, "provider unavailable"):
            OrderStructureAgent(_FailingModel()).extract(ORDERS[0])

    def test_unusable_model_outputs_are_rejected(self):
        invalid_cases = [
            ([], ORDERS[0], "JSON object"),
            (
                {**LABELED_OUTPUTS[ORDERS[0]], "confidence": "high"},
                ORDERS[0],
                "unexpected keys",
            ),
            (
                {**LABELED_OUTPUTS[ORDERS[0]], "buyer": "Invented Buyer"},
                ORDERS[0],
                "not grounded",
            ),
            (
                {
                    "order_id": "1001",
                    "buyer": "John Davis",
                    "state": "Ohio",
                    "total": "$742.10",
                },
                "Order 1001 Buyer John Davis State Ohio Total $742.10",
                "two-letter state code",
            ),
            (
                {
                    "order_id": "1001",
                    "buyer": "John Davis",
                    "state": "OH",
                    "total": "not-money",
                },
                "Order 1001 Buyer John Davis State OH Total not-money",
                "total.*invalid",
            ),
        ]

        for output, raw_order, message in invalid_cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    OrderStructureAgent(_FixedModel(output)).extract(raw_order)

    def test_openrouter_agent_can_be_configured(self):
        with patch.dict(
            os.environ,
            {"OPENROUTER_API_KEY": "configuration-test-only"},
        ):
            agent = create_openrouter_order_agent()

        self.assertIsInstance(agent, OrderStructureAgent)

if __name__ == "__main__":
    import unittest

    unittest.main()

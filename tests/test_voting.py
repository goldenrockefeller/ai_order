from __future__ import annotations

from unittest import TestCase

from database import Order
from order_extraction import OrderStructureAgent
from query_agent import FilterCondition, QueryAgent, QueryFilter
from fixtures.raw_orders import RAW_ORDERS as ORDERS
from voting import NoQuorumError, quorum_vote, validate_voting_config


ORDER_A = {
    "order_id": "1001",
    "buyer": "John Davis",
    "state": "OH",
    "total": "$742.10",
}
ORDER_B = {**ORDER_A, "buyer": "John"}

QUERY_A = {
    "logic": "all",
    "conditions": [
        {"field": "state", "operator": "eq", "values": ["OH"]},
        {"field": "total", "operator": "gt", "values": ["$500"]},
    ],
    "unsupported_conditions": [],
}
QUERY_B = {
    **QUERY_A,
    "conditions": [
        {"field": "state", "operator": "eq", "values": ["OH"]},
        {"field": "total", "operator": "gte", "values": ["$500"]},
    ],
}
QUERY_C = {
    **QUERY_A,
    "conditions": [
        {"field": "state", "operator": "eq", "values": ["TX"]},
        {"field": "total", "operator": "gt", "values": ["$500"]},
    ],
}


class _SequenceModel:
    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages)
        output = next(self.outputs)
        if isinstance(output, Exception):
            raise output
        return output


class VotingCorrectnessTests(TestCase):
    def test_a_a_b_accepts_a_and_reports_disagreement(self):
        values = iter(("A", "A", "B"))

        result = quorum_vote(
            lambda: next(values), attempts=3, quorum=2, label="test"
        )

        self.assertEqual(result.value, "A")
        self.assertTrue(result.disagreement)
        self.assertEqual(result.valid_votes, 3)
        self.assertEqual(result.rejected_votes, 0)

    def test_a_b_c_rejects_when_no_value_reaches_quorum(self):
        values = iter(("A", "B", "C"))

        result = quorum_vote(
            lambda: next(values), attempts=3, quorum=2, label="test"
        )

        self.assertIsNone(result.value)
        self.assertTrue(result.disagreement)

    def test_invalid_votes_and_model_failures_never_count(self):
        actions = iter(("A", ValueError("invalid"), RuntimeError("down"), "A"))

        def cast_vote():
            action = next(actions)
            if isinstance(action, Exception):
                raise action
            return action

        result = quorum_vote(
            cast_vote, attempts=4, quorum=2, label="test"
        )

        self.assertEqual(result.value, "A")
        self.assertEqual(result.valid_votes, 2)
        self.assertEqual(result.rejected_votes, 2)

    def test_required_configurations_use_the_configured_attempt_count(self):
        cases = ((1, 1, False), (3, 2, True), (5, 4, True))
        for attempts, quorum, disagreement in cases:
            with self.subTest(attempts=attempts, quorum=quorum):
                values = iter(("A", "A", "B", "A", "A"))
                result = quorum_vote(
                    lambda: next(values),
                    attempts=attempts,
                    quorum=quorum,
                    label="test",
                )
                self.assertEqual(result.value, "A")
                self.assertEqual(result.valid_votes, attempts)
                self.assertEqual(result.disagreement, disagreement)
                self.assertGreaterEqual(result.elapsed_seconds, 0)

    def test_invalid_voting_configuration_is_rejected(self):
        for attempts, quorum in ((0, 1), (1, 0), (2, 3), (True, 1)):
            with self.subTest(attempts=attempts, quorum=quorum):
                with self.assertRaises(ValueError):
                    validate_voting_config(attempts=attempts, quorum=quorum)

    def test_order_agent_votes_on_complete_normalized_records(self):
        model = _SequenceModel((ORDER_A, ORDER_A, ORDER_B))
        agent = OrderStructureAgent(model, attempts=3, quorum=2)

        result = agent.extract_vote(ORDERS[0])

        self.assertEqual(result.value, Order("1001", "John Davis", "OH", 74210))
        self.assertTrue(result.disagreement)
        self.assertEqual(len(model.calls), 3)
        self.assertEqual(len({call[0]["content"] for call in model.calls}), 3)
        self.assertEqual(model.calls[0][1], model.calls[1][1])
        self.assertEqual(model.calls[1][1], model.calls[2][1])

    def test_query_agent_uses_one_curated_prompt_per_vote(self):
        model = _SequenceModel((QUERY_A, QUERY_A, QUERY_A))

        result = QueryAgent(model, attempts=3, quorum=2).interpret_vote(
            "Ohio orders over $500"
        )

        self.assertIsNotNone(result.value)
        self.assertEqual(len({call[0]["content"] for call in model.calls}), 3)
        self.assertEqual(model.calls[0][1], model.calls[1][1])
        self.assertEqual(model.calls[1][1], model.calls[2][1])

    def test_agents_reject_more_votes_than_curated_prompt_variants(self):
        model = _SequenceModel(())

        with self.assertRaisesRegex(ValueError, "curated order prompt variants"):
            OrderStructureAgent(model, attempts=6, quorum=1)
        with self.assertRaisesRegex(ValueError, "curated query prompt variants"):
            QueryAgent(model, attempts=6, quorum=1)

    def test_query_agent_rejects_a_b_c_after_normalization(self):
        model = _SequenceModel((QUERY_A, QUERY_B, QUERY_C))
        agent = QueryAgent(model, attempts=3, quorum=2)

        result = agent.interpret_vote("Ohio orders over $500")

        self.assertIsNone(result.value)
        self.assertEqual(result.valid_votes, 3)
        with self.assertRaises(NoQuorumError):
            QueryAgent(
                _SequenceModel((QUERY_A, QUERY_B, QUERY_C)),
                attempts=3,
                quorum=2,
            ).interpret("Ohio orders over $500")

    def test_query_agent_rejects_invalid_output_before_counting_it(self):
        invalid = {**QUERY_A, "sql": "SELECT * FROM orders"}
        result = QueryAgent(
            _SequenceModel((QUERY_A, invalid, QUERY_A)),
            attempts=3,
            quorum=2,
        ).interpret_vote("Ohio orders over $500")

        self.assertEqual(
            result.value,
            QueryFilter(
                "all",
                (
                    FilterCondition("state", "eq", ("OH",)),
                    FilterCondition("total", "gt", ("$500",)),
                ),
            ),
        )
        self.assertEqual(result.valid_votes, 2)
        self.assertEqual(result.rejected_votes, 1)


if __name__ == "__main__":
    import unittest

    unittest.main()

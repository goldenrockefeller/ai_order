from __future__ import annotations

from unittest import TestCase

from query_agent import (
    FilterCondition,
    QueryAgent,
    QueryFilter,
    UnsupportedQueryError,
    normalize_query_output,
)


class _FixedModel:
    def __init__(self, output):
        self.output = output
        self.call_count = 0

    def invoke(self, _messages):
        self.call_count += 1
        return self.output


def _condition(field, operator, *values):
    return {"field": field, "operator": operator, "values": list(values)}


def _output(*conditions, logic="all", unsupported=None):
    return {
        "logic": logic,
        "conditions": list(conditions),
        "unsupported_conditions": unsupported or [],
    }


class QueryAgentCorrectnessTests(TestCase):
    def test_over_and_at_least_produce_different_operators(self):
        cases = (
            ("Ohio orders over $500", "gt"),
            ("Ohio orders at least $500", "gte"),
        )
        for request, operator in cases:
            with self.subTest(request=request):
                model = _FixedModel(
                    _output(
                        _condition("state", "eq", "OH"),
                        _condition("total", operator, "$500"),
                    )
                )
                actual = QueryAgent(model).interpret(request)
                self.assertEqual(
                    actual,
                    QueryFilter(
                        "all",
                        (
                            FilterCondition("state", "eq", ("OH",)),
                            FilterCondition("total", operator, ("$500",)),
                        ),
                    ),
                )
                self.assertEqual(model.call_count, 1)

    def test_compound_filter_supports_multiple_fields_and_membership(self):
        output = _output(
            _condition("state", "in", "OH", "TX"),
            _condition("buyer", "contains", "Ada"),
            _condition("total", "lt", "$1,000"),
        )

        self.assertEqual(
            normalize_query_output(output),
            QueryFilter(
                "all",
                (
                    FilterCondition("state", "in", ("OH", "TX")),
                    FilterCondition("buyer", "contains", ("Ada",)),
                    FilterCondition("total", "lt", ("$1,000",)),
                ),
            ),
        )

    def test_values_are_preserved_verbatim(self):
        result = normalize_query_output(
            _output(_condition("state", "eq", "Ohio"))
        )
        self.assertEqual(result.conditions[0].values, ("Ohio",))

    def test_unsupported_condition_is_rejected(self):
        with self.assertRaisesRegex(UnsupportedQueryError, "laptop"):
            normalize_query_output(
                _output(
                    _condition("state", "eq", "OH"),
                    unsupported=["contains item laptop"],
                )
            )

    def test_sql_or_other_extra_fields_are_rejected(self):
        output = _output(_condition("state", "eq", "OH"))
        output["sql"] = "SELECT * FROM orders"
        with self.assertRaisesRegex(ValueError, "unexpected keys: sql"):
            normalize_query_output(output)

    def test_operator_must_be_compatible_with_field(self):
        with self.assertRaisesRegex(ValueError, "unsupported for field 'buyer'"):
            normalize_query_output(
                _output(_condition("buyer", "gt", "Ada"))
            )

if __name__ == "__main__":
    import unittest

    unittest.main()

"""Small deterministic Checkpoint 6 comparison using the real agent boundaries."""

from __future__ import annotations

import json
import logging

from fixtures.raw_orders import RAW_ORDERS
from order_extraction import OrderStructureAgent
from query_agent import QueryAgent


ORDER_1001 = {
    "order_id": "1001",
    "buyer": "John Davis",
    "state": "OH",
    "total": "$742.10",
}
ORDER_1003 = {
    "order_id": "1003",
    "buyer": "Mike Turner",
    "state": "OH",
    "total": "$1299.99",
}
QUERY = {
    "logic": "all",
    "conditions": [
        {"field": "state", "operator": "eq", "values": ["OH"]},
        {"field": "total", "operator": "gt", "values": ["$500"]},
    ],
    "unsupported_conditions": [],
}


class _SequenceModel:
    def __init__(self, outputs: tuple[object, ...]) -> None:
        self._outputs = iter(outputs)

    def invoke(self, _messages):
        output = next(self._outputs)
        if isinstance(output, Exception):
            raise output
        return output


def _run_configuration(attempts: int, quorum: int) -> dict[str, object]:
    order_consensus = OrderStructureAgent(
        _SequenceModel(
            (
                ORDER_1001,
                ORDER_1001,
                {**ORDER_1001, "buyer": "John"},
                ORDER_1001,
                ORDER_1001,
            )
        ),
        attempts=attempts,
        quorum=quorum,
    ).extract_vote(RAW_ORDERS[0])
    order_split = OrderStructureAgent(
        _SequenceModel(
            (
                ORDER_1003,
                {**ORDER_1003, "buyer": "Mike"},
                {**ORDER_1003, "buyer": "Turner"},
                ORDER_1003,
                {**ORDER_1003, "buyer": "Mike"},
            )
        ),
        attempts=attempts,
        quorum=quorum,
    ).extract_vote(RAW_ORDERS[2])
    invalid_query = {**QUERY, "sql": "SELECT is forbidden"}
    query_with_invalid = QueryAgent(
        _SequenceModel((QUERY, invalid_query, QUERY, QUERY, QUERY)),
        attempts=attempts,
        quorum=quorum,
    ).interpret_vote("Ohio orders over $500")

    labeled_results = (
        ("order_consensus", order_consensus),
        ("order_split", order_split),
        ("query_with_invalid_vote", query_with_invalid),
    )
    return {
        "configuration": f"{quorum}-of-{attempts}",
        "accepted": [
            label for label, result in labeled_results if result.value is not None
        ],
        "skipped_or_rejected": [
            label for label, result in labeled_results if result.value is None
        ],
        "disagreements": [
            label for label, result in labeled_results if result.disagreement
        ],
        "rejected_votes": sum(
            result.rejected_votes for _, result in labeled_results
        ),
        "elapsed_model_call_ms": round(
            sum(result.elapsed_seconds for _, result in labeled_results) * 1000,
            3,
        ),
    }


def run_labeled_comparison() -> list[dict[str, object]]:
    """Run the same three labeled inputs under the required configurations."""

    return [
        _run_configuration(attempts, quorum)
        for attempts, quorum in ((1, 1), (3, 2), (5, 4))
    ]


def main() -> int:
    logging.basicConfig(level=logging.CRITICAL)
    print(json.dumps({"votingComparison": run_labeled_comparison()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

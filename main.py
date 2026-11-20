"""Checkpoint 5: run fetch, extract, store, and query through LangGraph."""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TypedDict

from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph

from database import Order, create_database, query_by_filter
from order_extraction import OrderStructureAgent, create_openrouter_order_agent
from query_agent import QueryAgent, create_openrouter_query_agent
from validated_raw_orders import (
    DEFAULT_API_URL,
    ValidatedRawOrders,
    fetch_validated_raw_orders,
)


logger = logging.getLogger(__name__)

DEFAULT_REQUEST = (
    "Show me all orders where the buyer was located in Ohio and total value "
    "was over 500."
)


@dataclass(frozen=True, slots=True)
class QueryResult:
    request: str
    orders: tuple[Order, ...]
    incomplete: bool
    failed_extractions: int


class PipelineState(TypedDict, total=False):
    requests: tuple[str, ...]
    raw_orders: tuple[str, ...]
    orders: tuple[Order, ...]
    failed_extractions: int
    connection: sqlite3.Connection
    results: tuple[QueryResult, ...]


def build_pipeline(
    *,
    order_agent: OrderStructureAgent,
    query_agent: QueryAgent,
    fetch_orders: Callable[[], ValidatedRawOrders],
):
    """Build the single-snapshot LangGraph with injectable external edges."""

    def fetch_node(_state: PipelineState) -> PipelineState:
        validated_orders = fetch_orders()
        return {"raw_orders": validated_orders.raw_orders}

    def extract_node(state: PipelineState) -> PipelineState:
        orders: list[Order] = []
        failed_extractions = 0
        for index, raw_order in enumerate(state["raw_orders"]):
            try:
                orders.append(order_agent.extract(raw_order))
            except Exception as error:
                failed_extractions += 1
                logger.error(
                    "Skipping failed order extraction: source_index=%d error=%s",
                    index,
                    error,
                )
        return {
            "orders": tuple(orders),
            "failed_extractions": failed_extractions,
        }

    def store_node(state: PipelineState) -> PipelineState:
        return {"connection": create_database(state["orders"])}

    def query_node(state: PipelineState) -> PipelineState:
        connection = state["connection"]
        try:
            results = tuple(
                QueryResult(
                    request=request,
                    orders=tuple(
                        query_by_filter(connection, query_agent.interpret(request))
                    ),
                    incomplete=state["failed_extractions"] > 0,
                    failed_extractions=state["failed_extractions"],
                )
                for request in state["requests"]
            )
        finally:
            connection.close()
        return {"results": results}

    graph = StateGraph(PipelineState)
    graph.add_node("fetch", fetch_node)
    graph.add_node("extract", extract_node)
    graph.add_node("store", store_node)
    graph.add_node("query", query_node)
    graph.add_edge(START, "fetch")
    graph.add_edge("fetch", "extract")
    graph.add_edge("extract", "store")
    graph.add_edge("store", "query")
    graph.add_edge("query", END)
    return graph.compile()


def run_pipeline(
    requests: Sequence[str],
    *,
    order_agent: OrderStructureAgent,
    query_agent: QueryAgent,
    fetch_orders: Callable[[], ValidatedRawOrders] = fetch_validated_raw_orders,
) -> tuple[QueryResult, ...]:
    """Run one ingestion snapshot and answer every supplied request from it."""

    normalized_requests = tuple(requests)
    if not normalized_requests:
        raise ValueError("at least one request is required")
    if any(
        not isinstance(request, str) or not request.strip()
        for request in normalized_requests
    ):
        raise ValueError("requests must contain non-empty strings")

    pipeline = build_pipeline(
        order_agent=order_agent,
        query_agent=query_agent,
        fetch_orders=fetch_orders,
    )
    final_state = pipeline.invoke({"requests": normalized_requests})
    return final_state["results"]


def _order_json(order: Order) -> dict[str, object]:
    return {
        "orderId": order.order_id,
        "buyer": order.buyer,
        "state": order.state,
        "total": order.total_cents / 100,
    }


def result_json(result: QueryResult) -> dict[str, object]:
    output: dict[str, object] = {
        "orders": [_order_json(order) for order in result.orders],
        "incomplete": result.incomplete,
    }
    if result.incomplete:
        output["failedExtractions"] = result.failed_extractions
    return output


def _output_json(results: tuple[QueryResult, ...]) -> dict[str, object]:
    if len(results) == 1:
        return result_json(results[0])
    return {
        "results": [
            {"request": result.request, **result_json(result)} for result in results
        ]
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Query one fetched and extracted customer-order snapshot."
    )
    parser.add_argument(
        "requests",
        nargs="*",
        help="Natural-language requests; quote each multi-word request.",
    )
    parser.add_argument(
        "--attempts",
        type=int,
        default=1,
        help="Independent model votes per order and query (default: 1).",
    )
    parser.add_argument(
        "--quorum",
        type=int,
        default=1,
        help="Matching normalized votes required (default: 1).",
    )
    arguments = parser.parse_args(argv)
    requests = arguments.requests or [DEFAULT_REQUEST]

    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    try:
        results = run_pipeline(
            requests,
            order_agent=create_openrouter_order_agent(
                attempts=arguments.attempts,
                quorum=arguments.quorum,
            ),
            query_agent=create_openrouter_query_agent(
                attempts=arguments.attempts,
                quorum=arguments.quorum,
            ),
            fetch_orders=lambda: fetch_validated_raw_orders(DEFAULT_API_URL),
        )
    except (
        ConnectionError,
        TimeoutError,
        RuntimeError,
        TypeError,
        ValueError,
        sqlite3.Error,
    ) as error:
        logger.error("Order pipeline failed: %s", error)
        return 1

    print(json.dumps(_output_json(results), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

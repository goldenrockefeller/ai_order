"""Checkpoint 3 demo: fetch a snapshot and structure every raw order once."""

from __future__ import annotations

import logging
import json

from dotenv import load_dotenv

from order_extraction import create_openrouter_order_agent
from snapshot import DEFAULT_API_URL, fetch_orders_snapshot


def _order_sort_key(order):
    if order.order_id.isascii() and order.order_id.isdecimal():
        return (0, int(order.order_id), order.order_id)
    return (1, 0, order.order_id)


def main() -> int:
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    try:
        snapshot = fetch_orders_snapshot(DEFAULT_API_URL)
    except (ConnectionError, TimeoutError, ValueError) as error:
        logging.error("Snapshot checkpoint failed: %s", error)
        return 1

    try:
        agent = create_openrouter_order_agent()
        orders = [agent.extract(raw_order) for raw_order in snapshot.raw_orders]
    except (RuntimeError, TypeError, ValueError) as error:
        logging.error("Order structure checkpoint failed: %s", error)
        return 1

    orders.sort(key=_order_sort_key)
    output = {
        "orders": [
            {
                "orderId": order.order_id,
                "buyer": order.buyer,
                "state": order.state,
                "total": order.total_cents / 100,
            }
            for order in orders
        ]
    }
    print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

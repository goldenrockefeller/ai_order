"""Deterministic, model-independent order filtering backed by SQLite."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Literal, Sequence


Operator = Literal["gt", "gte"]


@dataclass(frozen=True, slots=True)
class Order:
    """One normalized order row."""

    order_id: str
    buyer: str
    state: str
    total_cents: int


@dataclass(frozen=True, slots=True)
class OrderFilter:
    """The only filter shape accepted by :func:`query_orders`."""

    state: str
    total_cents: int
    operator: Operator

    def __post_init__(self) -> None:
        if not self.state:
            raise ValueError("state must not be empty")
        if isinstance(self.total_cents, bool) or not isinstance(self.total_cents, int):
            raise TypeError("total_cents must be an integer")
        if self.total_cents < 0:
            raise ValueError("total_cents must not be negative")
        if self.operator not in ("gt", "gte"):
            raise ValueError("operator must be 'gt' or 'gte'")


SEED_ORDERS: tuple[Order, ...] = (
    Order("1001", "John Davis", "OH", 74210),
    Order("1002", "Sarah Liu", "TX", 15655),
    Order("1003", "Mike Turner", "OH", 129999),
    Order("1004", "Rachel Kim", "WA", 8950),
    Order("1005", "Chris Myers", "OH", 51200),
)


def dollars_to_cents(amount: str | Decimal | int | float) -> int:
    """Convert a dollar amount to cents without binary floating-point math.

    Amounts must have no more than two fractional digits. Strings may include
    a leading ``$`` and commas for convenient handling of display values.
    """

    if isinstance(amount, bool) or not isinstance(amount, (str, Decimal, int, float)):
        raise TypeError("amount must be a string, Decimal, integer, or float")

    if isinstance(amount, str):
        text: str | Decimal | int = amount.strip().replace(",", "")
    elif isinstance(amount, float):
        # Convert through the float's shortest decimal representation instead
        # of importing its binary approximation into Decimal.
        text = str(amount)
    else:
        text = amount
    if isinstance(text, str) and text.startswith("$"):
        text = text[1:].strip()

    try:
        dollars = Decimal(text)
    except (InvalidOperation, ValueError) as error:
        raise ValueError(f"invalid dollar amount: {amount!r}") from error

    if not dollars.is_finite() or dollars < 0:
        raise ValueError("amount must be a finite, non-negative value")

    cents = dollars * 100
    if cents != cents.to_integral_value():
        raise ValueError("amount must have no more than two fractional digits")
    return int(cents)


def create_database() -> sqlite3.Connection:
    """Create and seed a new in-memory order database."""

    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    connection.executescript(
        """
        CREATE TABLE orders(
          order_id TEXT PRIMARY KEY,
          buyer TEXT NOT NULL,
          state TEXT NOT NULL,
          total_cents INTEGER NOT NULL
        );
        CREATE INDEX idx_orders_state_total
          ON orders(state, total_cents);
        """
    )
    insert_orders(connection, SEED_ORDERS)
    return connection


def insert_order(connection: sqlite3.Connection, order: Order) -> None:
    """Insert one normalized order using bound SQL parameters."""

    connection.execute(
        """
        INSERT INTO orders(order_id, buyer, state, total_cents)
        VALUES (?, ?, ?, ?)
        """,
        (order.order_id, order.buyer, order.state, order.total_cents),
    )
    connection.commit()


def insert_orders(
    connection: sqlite3.Connection, orders: Sequence[Order]
) -> None:
    """Insert normalized orders using bound SQL parameters."""

    connection.executemany(
        """
        INSERT INTO orders(order_id, buyer, state, total_cents)
        VALUES (?, ?, ?, ?)
        """,
        ((order.order_id, order.buyer, order.state, order.total_cents) for order in orders),
    )
    connection.commit()


def query_orders(
    connection: sqlite3.Connection, order_filter: OrderFilter
) -> list[Order]:
    """Return matching orders in deterministic numeric order-id order.

    The caller supplies typed values only. The comparison operator is selected
    from a fixed internal mapping and never interpolated from caller input.
    """

    query = {
        "gt": """
            SELECT order_id, buyer, state, total_cents
            FROM orders
            WHERE state = ? AND total_cents > ?
            ORDER BY CAST(order_id AS INTEGER), order_id
        """,
        "gte": """
            SELECT order_id, buyer, state, total_cents
            FROM orders
            WHERE state = ? AND total_cents >= ?
            ORDER BY CAST(order_id AS INTEGER), order_id
        """,
    }[order_filter.operator]
    rows = connection.execute(
        query, (order_filter.state, order_filter.total_cents)
    ).fetchall()
    return [
        Order(
            order_id=row["order_id"],
            buyer=row["buyer"],
            state=row["state"],
            total_cents=row["total_cents"],
        )
        for row in rows
    ]


__all__ = [
    "Order",
    "OrderFilter",
    "SEED_ORDERS",
    "create_database",
    "dollars_to_cents",
    "insert_order",
    "insert_orders",
    "query_orders",
]

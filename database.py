"""Deterministic, model-independent order filtering backed by SQLite."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import TYPE_CHECKING, Any, Literal, Sequence

if TYPE_CHECKING:
    from query_agent import FilterCondition, QueryFilter


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


def create_database(
    orders: Sequence[Order] = (),
) -> sqlite3.Connection:
    """Create an empty in-memory database and insert explicit ``orders``."""

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
    insert_orders(connection, orders)
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


_COLUMNS = {
    "order_id": "order_id",
    "buyer": "buyer",
    "state": "state",
    "total": "total_cents",
}
_SINGLE_VALUE_OPERATORS = {
    "eq": "=",
    "neq": "!=",
    "gt": ">",
    "gte": ">=",
    "lt": "<",
    "lte": "<=",
}


def _condition_values(condition: "FilterCondition") -> tuple[Any, ...]:
    """Parse model-preserved values at the database execution boundary."""

    values = condition.values
    if not values or any(not isinstance(value, str) or not value.strip() for value in values):
        raise ValueError("filter values must contain non-empty strings")

    if condition.field == "total":
        return tuple(dollars_to_cents(value) for value in values)
    if condition.field == "state":
        normalized: list[str] = []
        for value in values:
            if len(value) != 2 or not value.isascii() or not value.isalpha():
                raise ValueError("state filter values must be two-letter state codes")
            normalized.append(value.upper())
        return tuple(normalized)
    return tuple(values)


def _compile_condition(condition: "FilterCondition") -> tuple[str, tuple[Any, ...]]:
    """Compile one typed condition from fixed SQL fragments and bound values."""

    try:
        column = _COLUMNS[condition.field]
    except KeyError as error:
        raise ValueError(f"unsupported filter field: {condition.field!r}") from error

    values = _condition_values(condition)
    if condition.operator == "contains":
        if len(values) != 1 or condition.field not in ("order_id", "buyer"):
            raise ValueError("contains requires one order_id or buyer value")
        return f"instr({column}, ?) > 0", values

    if condition.operator in ("in", "not_in"):
        placeholders = ", ".join("?" for _ in values)
        sql_operator = "IN" if condition.operator == "in" else "NOT IN"
        return f"{column} {sql_operator} ({placeholders})", values

    try:
        sql_operator = _SINGLE_VALUE_OPERATORS[condition.operator]
    except KeyError as error:
        raise ValueError(
            f"unsupported filter operator: {condition.operator!r}"
        ) from error
    if len(values) != 1:
        raise ValueError(f"operator {condition.operator!r} requires one value")
    if condition.field != "total" and condition.operator in ("gt", "gte", "lt", "lte"):
        raise ValueError(
            f"operator {condition.operator!r} is unsupported for field {condition.field!r}"
        )
    return f"{column} {sql_operator} ?", values


def query_by_filter(
    connection: sqlite3.Connection, query_filter: "QueryFilter"
) -> list[Order]:
    """Execute a compound typed filter without accepting model-written SQL."""

    if query_filter.logic not in ("all", "any"):
        raise ValueError("filter logic must be 'all' or 'any'")
    if not query_filter.conditions:
        raise ValueError("filter must contain at least one condition")

    clauses: list[str] = []
    parameters: list[Any] = []
    for condition in query_filter.conditions:
        clause, values = _compile_condition(condition)
        clauses.append(clause)
        parameters.extend(values)

    joiner = " AND " if query_filter.logic == "all" else " OR "
    statement = f"""
        SELECT order_id, buyer, state, total_cents
        FROM orders
        WHERE {joiner.join(clauses)}
        ORDER BY CAST(order_id AS INTEGER), order_id
    """
    rows = connection.execute(statement, parameters).fetchall()
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
    "create_database",
    "dollars_to_cents",
    "insert_order",
    "insert_orders",
    "query_by_filter",
    "query_orders",
]

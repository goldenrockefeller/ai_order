from __future__ import annotations

from unittest import TestCase

from database import (
    Order,
    OrderFilter,
    create_database,
    dollars_to_cents,
    insert_order,
    query_orders,
)
from tests.fixtures.labeled_orders import LABELED_ORDERS


class DatabaseCorrectnessTests(TestCase):
    def setUp(self):
        self.connection = create_database(LABELED_ORDERS)

    def tearDown(self):
        self.connection.close()

    def test_database_is_empty_without_explicit_orders(self):
        empty_connection = create_database()
        try:
            matches = query_orders(
                empty_connection, OrderFilter("OH", 0, "gte")
            )
        finally:
            empty_connection.close()

        self.assertEqual(matches, [])

    def test_oh_orders_over_500_dollars(self):
        matches = query_orders(
            self.connection,
            OrderFilter(
                state="OH",
                total_cents=dollars_to_cents("500.00"),
                operator="gt",
            ),
        )

        self.assertEqual(
            [order.order_id for order in matches], ["1001", "1003", "1005"]
        )

    def test_greater_than_excludes_boundary_and_greater_than_or_equal_includes_it(self):
        insert_order(self.connection, Order("1006", "Boundary Buyer", "OH", 50000))

        greater_than = query_orders(
            self.connection, OrderFilter("OH", 50000, "gt")
        )
        greater_than_or_equal = query_orders(
            self.connection, OrderFilter("OH", 50000, "gte")
        )

        self.assertNotIn("1006", [order.order_id for order in greater_than])
        self.assertIn("1006", [order.order_id for order in greater_than_or_equal])

if __name__ == "__main__":
    import unittest

    unittest.main()

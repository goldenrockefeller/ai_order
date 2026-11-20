"""Normalized labels for database and pipeline correctness tests."""

from database import Order


LABELED_ORDERS = (
    Order("1001", "John Davis", "OH", 74210),
    Order("1002", "Sarah Liu", "TX", 15655),
    Order("1003", "Mike Turner", "OH", 129999),
    Order("1004", "Rachel Kim", "WA", 8950),
    Order("1005", "Chris Myers", "OH", 51200),
)

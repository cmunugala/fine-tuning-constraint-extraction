"""Shared generation vocabulary and split policies."""

from __future__ import annotations

CONSTRAINT_TYPES = (
    "unavailable",
    "available",
    "preferred_hours",
    "min_weekly_hours",
    "max_weekly_hours",
    "max_consecutive_days",
    "min_rest_hours",
)

TYPE_WEIGHTS = (
    22,
    16,
    18,
    12,
    14,
    9,
    9,
)

TEMPORAL_TYPES = frozenset({"unavailable", "available", "preferred_hours"})
WEEKLY_TYPES = frozenset({"min_weekly_hours", "max_weekly_hours"})

NAMES = (
    "Aisha", "Alejandro", "Amara", "Ben", "Camille", "Chen", "Daniel",
    "Elena", "Fatima", "Grace", "Haruto", "Imani", "Iris", "Jamal",
    "Jordan", "Kai", "Leila", "Liam", "Maya", "Mateo", "Nadia", "Noah",
    "Priya", "Ravi", "Ren", "Sam", "Sofia", "Tariq", "Taylor", "Zoe",
)

DAYS = (
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday",
    "sunday",
)

# These combinations may occur in train and validation. Test also uses them to
# measure phrasing generalization independently of compositional generalization.
SEEN_COMBINATIONS = (
    ("unavailable", "max_weekly_hours"),
    ("available", "preferred_hours"),
    ("preferred_hours", "min_weekly_hours"),
    ("min_weekly_hours", "max_weekly_hours"),
    ("max_consecutive_days", "min_rest_hours"),
    ("unavailable", "preferred_hours"),
    ("unavailable", "max_weekly_hours", "min_rest_hours"),
    ("available", "preferred_hours", "max_weekly_hours"),
    ("min_weekly_hours", "max_weekly_hours", "max_consecutive_days"),
    ("unavailable", "max_consecutive_days", "min_rest_hours"),
)

# Exactly these unordered type sets are reserved from train and validation.
HELD_OUT_COMBINATIONS = (
    ("available", "min_weekly_hours"),
    ("unavailable", "min_rest_hours"),
    ("preferred_hours", "max_weekly_hours", "max_consecutive_days"),
)


def combination_key(types: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    """Return an order-independent key for a constraint-type combination."""

    return tuple(sorted(types))


HELD_OUT_KEYS = frozenset(combination_key(item) for item in HELD_OUT_COMBINATIONS)

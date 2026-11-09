"""Shared generation vocabulary and split policies."""

from __future__ import annotations

from itertools import combinations

CONSTRAINT_TYPES = (
    "unavailable",
    "available",
    "preferred_hours",
    "min_weekly_hours",
    "max_weekly_hours",
    "max_consecutive_days",
    "min_rest_hours",
)

TEMPORAL_TYPES = frozenset({"unavailable", "available", "preferred_hours"})
WEEKLY_TYPES = frozenset({"min_weekly_hours", "max_weekly_hours"})

NAMES = (
    "Aisha", "Alejandro", "Amara", "Ben", "Camille", "Chen", "Daniel",
    "Elena", "Fatima", "Grace", "Haruto", "Imani", "Iris", "Jamal",
    "Jordan", "Kai", "Leila", "Liam", "Maya", "Mateo", "Nadia", "Noah",
    "Priya", "Ravi", "Ren", "Sam", "Sofia", "Tariq", "Taylor", "Zoe",
    "Abdul", "Ada", "Aiden", "Ana", "Anika", "Arjun", "Beatriz", "Chloe",
    "Darius", "Diego", "Eli", "Emi", "Farah", "Gabriel", "Hana", "Hugo",
    "Ines", "Jae", "Jin", "Jo", "José", "Keiko", "Lena", "Luis", "Mina",
    "Miriam", "Nikhil", "Omar", "Parker", "Quinn", "Rosa", "Saanvi",
    "Theo", "Uma", "Val", "Wei", "Yara", "Yuki", "Zain", "Zoë",
    "Anne-Marie", "D'Andre", "Jean-Luc", "Maryam", "Nia", "O'Neil",
)

DAYS = (
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday",
    "sunday",
)

# The pooled corpus cycles through every distinct pair and triple before the
# completed records are randomly split. This avoids the old fixed combination
# list and gives every split a chance to contain the same semantic structures.
PAIR_COMBINATIONS = tuple(combinations(CONSTRAINT_TYPES, 2))
TRIPLE_COMBINATIONS = tuple(combinations(CONSTRAINT_TYPES, 3))


def combination_key(types: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    """Return an order-independent key for a constraint-type combination."""

    return tuple(sorted(types))

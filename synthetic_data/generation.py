"""Ground-truth-first scenario generation using the project's Pydantic models."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, time, timedelta
import random
from typing import Literal

from models import SchedulingConstraint, SchedulingOutput, TimeScope

from .catalog import DAYS, NAMES, TEMPORAL_TYPES

ScopeStyle = Literal["recurring", "one_off", "bounded"]


@dataclass(frozen=True)
class Scenario:
    """A generated label plus context needed to verbalize it."""

    output: SchedulingOutput
    reference_date: date


def random_reference_date(rng: random.Random) -> date:
    start = date(2025, 1, 1)
    return start + timedelta(days=rng.randrange((date(2027, 12, 31) - start).days + 1))


def choose_priority(constraint_type: str, rng: random.Random) -> str:
    if constraint_type == "preferred_hours":
        return "soft"
    soft_probability = {
        "unavailable": 0.24,
        "available": 0.18,
        "min_weekly_hours": 0.15,
        "max_weekly_hours": 0.18,
        "max_consecutive_days": 0.08,
        "min_rest_hours": 0.08,
    }[constraint_type]
    return "soft" if rng.random() < soft_probability else "hard"


def _time_scope(
    constraint_type: str,
    reference_date: date,
    rng: random.Random,
    *,
    style: ScopeStyle | None = None,
    forced_day: str | None = None,
    one_off_offset: int | None = None,
) -> TimeScope:
    style = style or rng.choices(
        ["recurring", "one_off", "bounded"], weights=[55, 25, 20], k=1
    )[0]

    days: list[str] | None = None
    start_date: date | None = None
    end_date: date | None = None
    recurring = style != "one_off"

    if recurring:
        first_day = forced_day or rng.choice(DAYS)
        days = [first_day]
        if rng.random() < 0.22:
            second = rng.choice([day for day in DAYS if day != first_day])
            days.append(second)
            days.sort(key=DAYS.index)
        if style == "bounded":
            start_date = reference_date + timedelta(days=rng.randint(1, 14))
            end_date = start_date + timedelta(days=rng.randint(28, 84))
    else:
        start_date = reference_date + timedelta(
            days=one_off_offset if one_off_offset is not None else rng.randint(1, 28)
        )
        end_date = start_date

    start_time: time | None = None
    end_time: time | None = None
    if constraint_type in {"available", "preferred_hours"}:
        start_hour = rng.choice([6, 7, 8, 9, 10, 11, 12, 13, 14])
        duration = rng.choice([3, 4, 5, 6, 7, 8])
        start_time = time(start_hour)
        end_time = time(min(start_hour + duration, 22))
    else:
        time_pattern = rng.choices(
            ["full_day", "after", "before", "range"],
            weights=[28, 28, 20, 24],
            k=1,
        )[0]
        if time_pattern == "after":
            start_time = time(rng.choice([12, 14, 15, 16, 17, 18, 19]))
        elif time_pattern == "before":
            end_time = time(rng.choice([8, 9, 10, 11, 12, 13]))
        elif time_pattern == "range":
            start_hour = rng.choice([6, 7, 8, 9, 10, 12, 14, 16])
            start_time = time(start_hour)
            end_time = time(min(start_hour + rng.choice([3, 4, 5, 6]), 22))

    return TimeScope(
        days_of_week=days,
        start_time=start_time,
        end_time=end_time,
        start_date=start_date,
        end_date=end_date,
        recurring=recurring,
    )


def make_constraint(
    constraint_type: str,
    employee: str | None,
    reference_date: date,
    rng: random.Random,
    *,
    priority: str | None = None,
    scope_style: ScopeStyle | None = None,
    forced_day: str | None = None,
    one_off_offset: int | None = None,
    value: float | None = None,
) -> SchedulingConstraint:
    """Create one semantically meaningful constraint supported by the schema."""

    priority = priority or choose_priority(constraint_type, rng)
    if constraint_type in TEMPORAL_TYPES:
        return SchedulingConstraint(
            employee=employee,
            type=constraint_type,
            priority=priority,
            time_scope=_time_scope(
                constraint_type,
                reference_date,
                rng,
                style=scope_style,
                forced_day=forced_day,
                one_off_offset=one_off_offset,
            ),
        )

    if value is None:
        choices = {
            "min_weekly_hours": [12, 16, 20, 24, 28, 30, 32],
            "max_weekly_hours": [20, 24, 28, 30, 32, 36, 40],
            "max_consecutive_days": [2, 3, 4, 5, 6],
            "min_rest_hours": [8, 10, 11, 12, 14, 16],
        }
        value = float(rng.choice(choices[constraint_type]))
    return SchedulingConstraint(
        employee=employee,
        type=constraint_type,
        priority=priority,
        value=value,
    )


def make_scenario(
    constraint_types: tuple[str, ...],
    rng: random.Random,
    *,
    reference_date: date | None = None,
    team_wide: bool = False,
    multiple_employees: bool = False,
    scope_styles: tuple[ScopeStyle | None, ...] | None = None,
    priorities: tuple[str | None, ...] | None = None,
    one_off_offsets: tuple[int | None, ...] | None = None,
) -> Scenario:
    """Generate a compatible single- or multi-constraint scenario."""

    reference_date = reference_date or random_reference_date(rng)
    if team_wide:
        employees: list[str | None] = [None] * len(constraint_types)
    elif multiple_employees and len(constraint_types) > 1:
        employees = list(rng.sample(NAMES, k=len(constraint_types)))
    else:
        employee = rng.choice(NAMES)
        employees = [employee] * len(constraint_types)

    styles = scope_styles or tuple(None for _ in constraint_types)
    requested_priorities = priorities or tuple(None for _ in constraint_types)
    offsets = one_off_offsets or tuple(None for _ in constraint_types)
    temporal_index = 0
    base_day_index = rng.randrange(len(DAYS))
    constraints: list[SchedulingConstraint] = []

    min_hours: float | None = None
    max_hours: float | None = None
    if "min_weekly_hours" in constraint_types and "max_weekly_hours" in constraint_types:
        min_hours = float(rng.choice([12, 16, 20, 24, 28]))
        max_hours = float(rng.choice([value for value in [24, 28, 30, 32, 36, 40] if value > min_hours]))

    for index, constraint_type in enumerate(constraint_types):
        forced_day = None
        if constraint_type in TEMPORAL_TYPES:
            # Separate temporal rules within one request to avoid accidental conflicts.
            forced_day = DAYS[(base_day_index + temporal_index * 2) % len(DAYS)]
            temporal_index += 1
        forced_value = None
        if constraint_type == "min_weekly_hours":
            forced_value = min_hours
        elif constraint_type == "max_weekly_hours":
            forced_value = max_hours
        constraints.append(
            make_constraint(
                constraint_type,
                employees[index],
                reference_date,
                rng,
                priority=requested_priorities[index],
                scope_style=styles[index],
                forced_day=forced_day,
                one_off_offset=offsets[index],
                value=forced_value,
            )
        )

    return Scenario(SchedulingOutput(constraints=constraints), reference_date)

from pydantic import BaseModel, Field
from typing import Literal
from datetime import date, time


class TimeScope(BaseModel):
    days_of_week: list[
        Literal[
            "monday", "tuesday", "wednesday",
            "thursday", "friday", "saturday", "sunday"
        ]
    ] | None = None

    start_time: time | None = None
    end_time: time | None = None

    start_date: date | None = None
    end_date: date | None = None

    recurring: bool = False


class SchedulingConstraint(BaseModel):
    employee: str | None = None

    type: Literal[
        "unavailable",
        "available",
        "preferred_hours",
        "min_weekly_hours",
        "max_weekly_hours",
        "max_consecutive_days",
        "min_rest_hours"
    ]

    priority: Literal["hard", "soft"] = "hard"

    value: float | None = None

    time_scope: TimeScope | None = None


class SchedulingOutput(BaseModel):
    constraints: list[SchedulingConstraint]
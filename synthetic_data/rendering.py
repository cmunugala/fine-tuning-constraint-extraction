"""Split-specific natural-language realization of generated constraints."""

from __future__ import annotations

from datetime import date, time
import random

from models import SchedulingConstraint, TimeScope

from .generation import Scenario


TEMPLATES: dict[str, dict[str, dict[str, tuple[str, ...]]]] = {
    "train": {
        "unavailable": {
            "hard": (
                "{subject} can't work {scope}",
                "Do not schedule {subject} {scope}",
                "{scope} does not work for {subject}",
            ),
            "soft": (
                "{subject} would rather not work {scope}",
                "If possible, avoid scheduling {subject} {scope}",
            ),
        },
        "available": {
            "hard": (
                "{subject} is only available {scope}",
                "Only schedule {subject} {scope}",
                "{subject} can take shifts {scope}",
            ),
            "soft": (
                "{subject} would like to work {scope} if possible",
                "Try to schedule {subject} {scope}",
            ),
        },
        "preferred_hours": {
            "soft": (
                "{subject} prefers working {scope}",
                "{subject}'s preferred shift window is {scope}",
                "When possible, give {subject} shifts {scope}",
            ),
        },
        "min_weekly_hours": {
            "hard": (
                "{subject} needs at least {value} hours each week",
                "Schedule {subject} for a minimum of {value} hours per week",
            ),
            "soft": (
                "Aim to give {subject} at least {value} hours a week",
                "Ideally, {subject} should get {value} or more hours weekly",
            ),
        },
        "max_weekly_hours": {
            "hard": (
                "Keep {subject} at or below {value} hours per week",
                "{subject} must not exceed {value} hours in a week",
            ),
            "soft": (
                "Try not to give {subject} more than {value} hours weekly",
                "Preferably keep {subject} under {value} hours a week",
            ),
        },
        "max_consecutive_days": {
            "hard": (
                "{subject} can work no more than {value} days in a row",
                "Limit {subject} to {value} consecutive workdays",
            ),
            "soft": ("Try to avoid scheduling {subject} more than {value} days straight",),
        },
        "min_rest_hours": {
            "hard": (
                "Give {subject} at least {value} hours between shifts",
                "{subject} needs a minimum {value}-hour break between shifts",
            ),
            "soft": ("Try to leave {value} hours of rest between {subject}'s shifts",),
        },
    },
    "validation": {
        "unavailable": {
            "hard": (
                "Keep {subject} off the rota {scope}",
                "{subject} is unavailable {scope}",
            ),
            "soft": ("Please avoid putting {subject} on the rota {scope} when you can",),
        },
        "available": {
            "hard": (
                "{subject}'s availability is limited to {scope}",
                "Book {subject} only {scope}",
            ),
            "soft": ("It would be best to place {subject} {scope}",),
        },
        "preferred_hours": {
            "soft": (
                "{subject} has asked for shifts {scope}",
                "Please favor {scope} for {subject}",
            ),
        },
        "min_weekly_hours": {
            "hard": ("Make sure {subject} receives no fewer than {value} weekly hours",),
            "soft": ("Please target at least {value} hours a week for {subject}",),
        },
        "max_weekly_hours": {
            "hard": ("Cap {subject}'s weekly schedule at {value} hours",),
            "soft": ("The target ceiling for {subject} is {value} hours per week",),
        },
        "max_consecutive_days": {
            "hard": ("After {value} straight workdays, {subject} must have a day off",),
            "soft": ("Please aim for no runs longer than {value} days for {subject}",),
        },
        "min_rest_hours": {
            "hard": ("Leave a gap of at least {value} hours between shifts for {subject}",),
            "soft": ("A {value}-hour turnaround between {subject}'s shifts is preferred",),
        },
    },
    "test": {
        "unavailable": {
            "hard": (
                "{subject} is not able to take shifts {scope}",
                "Block out {scope} for {subject}",
            ),
            "soft": ("Where feasible, leave {subject} unscheduled {scope}",),
        },
        "available": {
            "hard": (
                "The only workable slot for {subject} is {scope}",
                "{subject} may be rostered {scope} and not outside that window",
            ),
            "soft": ("{scope} is the best scheduling window for {subject}",),
        },
        "preferred_hours": {
            "soft": (
                "{subject} has a strong preference for {scope}",
                "Favor shifts {scope} when assigning {subject}",
            ),
        },
        "min_weekly_hours": {
            "hard": ("Do not let {subject}'s weekly total fall below {value} hours",),
            "soft": ("A weekly total of at least {value} hours is preferred for {subject}",),
        },
        "max_weekly_hours": {
            "hard": ("Do not roster {subject} beyond {value} total hours each week",),
            "soft": ("A weekly load over {value} hours should be avoided for {subject}",),
        },
        "max_consecutive_days": {
            "hard": ("A run of shifts for {subject} must end after {value} days",),
            "soft": ("Whenever feasible, break up {subject}'s schedule after {value} days",),
        },
        "min_rest_hours": {
            "hard": ("There must be {value} hours of downtime before {subject}'s next shift",),
            "soft": ("Allowing {subject} a {value}-hour rest window is preferred",),
        },
    },
    "preview": {
        "unavailable": {
            "hard": ("{subject} is unable to work {scope}",),
            "soft": ("Please try to keep {subject} free {scope}",),
        },
        "available": {
            "hard": ("{subject} is available only {scope}",),
            "soft": ("{subject} would like to be scheduled {scope}",),
        },
        "preferred_hours": {
            "soft": ("{subject} would prefer shifts {scope}",),
        },
        "min_weekly_hours": {
            "hard": ("Please give {subject} at least {value} hours per week",),
            "soft": ("A minimum of {value} weekly hours is preferred for {subject}",),
        },
        "max_weekly_hours": {
            "hard": ("Please keep {subject} to no more than {value} hours per week",),
            "soft": ("It would be best to keep {subject} below {value} weekly hours",),
        },
        "max_consecutive_days": {
            "hard": ("Do not schedule {subject} for more than {value} days in succession",),
            "soft": ("A run longer than {value} days should be avoided for {subject}",),
        },
        "min_rest_hours": {
            "hard": ("Allow at least {value} hours of rest between shifts for {subject}",),
            "soft": ("A rest period of {value} hours is preferred for {subject}",),
        },
    },
}

CONNECTORS = {
    "train": (". Also, ", "; additionally, ", ". One more thing: "),
    "validation": (". Please also note that ", "; at the same time, "),
    "test": (". On top of that, ", "; another requirement is that "),
    "preview": (". The same request also says: ", "; plus, "),
}


def assert_template_banks_disjoint() -> None:
    """Fail if an exact clause or connector template is assigned to two splits."""

    owners: dict[str, str] = {}
    for split, by_type in TEMPLATES.items():
        phrases = {
            template
            for by_priority in by_type.values()
            for variants in by_priority.values()
            for template in variants
        }
        phrases.update(CONNECTORS[split])
        for phrase in phrases:
            previous = owners.setdefault(phrase, split)
            if previous != split:
                raise ValueError(
                    f"Phrasing template is shared by {previous} and {split}: {phrase!r}"
                )


def _subject(constraint: SchedulingConstraint) -> str:
    return constraint.employee if constraint.employee is not None else "each team member"


def _format_number(value: float | None) -> str:
    if value is None:
        raise ValueError("Numeric constraint is missing its value")
    return str(int(value)) if value.is_integer() else str(value)


def _clock(value: time, split: str) -> str:
    if split == "test":
        return value.strftime("%H:%M")
    hour = value.hour % 12 or 12
    minute = f":{value.minute:02d}" if value.minute else ""
    suffix = "am" if value.hour < 12 else "pm"
    if split == "validation":
        suffix = suffix.upper()
    if value.hour == 12 and value.minute == 0 and split == "train":
        return "noon"
    return f"{hour}{minute} {suffix}"


def _calendar_day(
    value: date,
    reference_date: date,
    split: str,
    *,
    allow_relative: bool = True,
) -> tuple[str, str]:
    delta = (value - reference_date).days
    if allow_relative and delta == 1:
        variants = {
            "train": ("tomorrow", "date-relative-tomorrow"),
            "validation": ("the following day", "date-relative-following"),
            "test": ("in one day", "date-relative-one-day"),
            "preview": ("tomorrow", "date-relative-preview"),
        }
        return variants[split]
    if allow_relative and 2 <= delta <= 7:
        variants = {
            "train": (f"this coming {value.strftime('%A')}", "date-relative-coming"),
            "validation": (f"in {delta} days", "date-relative-countdown"),
            "test": (f"next {value.strftime('%A')}", "date-relative-next"),
            "preview": (f"in {delta} days", "date-relative-preview-count"),
        }
        return variants[split]
    formats = {
        "train": (value.strftime("%B %d").replace(" 0", " "), "date-month-name"),
        "validation": (value.strftime("%b %d, %Y").replace(" 0", " "), "date-month-year"),
        "test": (value.isoformat(), "date-iso"),
        "preview": (value.strftime("%B %d").replace(" 0", " "), "date-preview"),
    }
    return formats[split]


def _day_list(days: list[str]) -> str:
    title_days = [day.capitalize() for day in days]
    if len(title_days) == 1:
        return title_days[0]
    return f"{', '.join(title_days[:-1])} and {title_days[-1]}"


def _scope(scope: TimeScope, reference_date: date, split: str) -> tuple[str, list[str]]:
    parts: list[str] = []
    ids: list[str] = []
    if scope.recurring:
        if not scope.days_of_week:
            raise ValueError("Recurring scope is missing weekdays")
        prefix = "every" if len(scope.days_of_week) == 1 else "each"
        parts.append(f"{prefix} {_day_list(scope.days_of_week)}")
        ids.append(f"{split}:scope-recurring")
        if scope.start_date and scope.end_date:
            start, start_id = _calendar_day(
                scope.start_date, reference_date, split, allow_relative=False
            )
            end, end_id = _calendar_day(
                scope.end_date, reference_date, split, allow_relative=False
            )
            parts.append(f"between {start} and {end}")
            ids.extend((f"{split}:{start_id}", f"{split}:{end_id}"))
    elif scope.start_date:
        day, date_id = _calendar_day(scope.start_date, reference_date, split)
        needs_on = not day.startswith(("tomorrow", "the following", "in ", "next "))
        parts.append(f"on {day}" if needs_on else day)
        ids.append(f"{split}:{date_id}")

    if scope.start_time and scope.end_time:
        parts.append(
            f"from {_clock(scope.start_time, split)} to {_clock(scope.end_time, split)}"
        )
        ids.append(f"{split}:scope-time-range")
    elif scope.start_time:
        parts.append(f"after {_clock(scope.start_time, split)}")
        ids.append(f"{split}:scope-after")
    elif scope.end_time:
        parts.append(f"before {_clock(scope.end_time, split)}")
        ids.append(f"{split}:scope-before")

    if not parts:
        raise ValueError("Temporal constraint has an empty scope")
    return " ".join(parts), ids


def render_scenario(
    scenario: Scenario, split: str, rng: random.Random
) -> tuple[str, tuple[str, ...]]:
    clauses: list[str] = []
    template_ids: list[str] = []
    for constraint in scenario.output.constraints:
        priority_templates = TEMPLATES[split][constraint.type]
        variants = priority_templates[constraint.priority]
        variant_index = rng.randrange(len(variants))
        template = variants[variant_index]
        scope_text = ""
        if constraint.time_scope is not None:
            scope_text, scope_ids = _scope(
                constraint.time_scope, scenario.reference_date, split
            )
            template_ids.extend(scope_ids)
        clause = template.format(
            subject=_subject(constraint),
            scope=scope_text,
            value=_format_number(constraint.value) if constraint.value is not None else "",
        )
        clauses.append(clause[0].upper() + clause[1:])
        template_ids.append(
            f"{split}:clause:{constraint.type}:{constraint.priority}:{variant_index}"
        )

    text = clauses[0]
    connector_offset = rng.randrange(len(CONNECTORS[split])) if len(clauses) > 1 else 0
    for index, clause in enumerate(clauses[1:]):
        connector_index = (connector_offset + index) % len(CONNECTORS[split])
        connector = CONNECTORS[split][connector_index]
        if text.endswith(".") and connector.startswith("."):
            connector = connector[1:]
        next_constraint = scenario.output.constraints[index + 1]
        if next_constraint.employee and clause.startswith(next_constraint.employee):
            joined_clause = clause
        else:
            joined_clause = clause[0].lower() + clause[1:]
        text += connector + joined_clause
        template_ids.append(f"{split}:connector:{connector_index}")
    return text.rstrip(".") + ".", tuple(template_ids)

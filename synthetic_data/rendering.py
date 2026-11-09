"""Diverse, split-agnostic natural-language realization of scenarios."""

from __future__ import annotations

from datetime import date, time
import random

from models import SchedulingConstraint, TimeScope

from .generation import Scenario


STYLES = ("conversational", "direct", "formal", "terse")
STYLE_WEIGHTS = (35, 25, 20, 20)

# Scope, clock, date, opener, connector, and light-noise choices compose with
# these register-specific clauses, creating hundreds of distinct structures.
CLAUSE_TEMPLATES: dict[str, dict[str, dict[str, tuple[str, ...]]]] = {
    "unavailable": {
        "hard": {
            "conversational": (
                "{subject} can't work {scope}",
                "{subject} isn't able to take shifts {scope}",
            ),
            "direct": (
                "Do not schedule {subject} {scope}",
                "Keep {subject} off the rota {scope}",
            ),
            "formal": (
                "{subject} is unavailable {scope}",
                "Shifts for {subject} are prohibited {scope}",
            ),
            "terse": (
                "No shifts for {subject} {scope}",
                "{subject}: unavailable {scope}",
            ),
        },
        "soft": {
            "conversational": (
                "{subject} would rather not work {scope}",
                "If you can, avoid giving {subject} shifts {scope}",
            ),
            "direct": (
                "Try not to schedule {subject} {scope}",
                "Please avoid assigning {subject} {scope} when possible",
            ),
            "formal": (
                "{subject} has requested not to work {scope}, if feasible",
                "Shifts {scope} should preferably be avoided for {subject}",
            ),
            "terse": (
                "Prefer no shifts for {subject} {scope}",
                "{subject}: avoid {scope} if possible",
            ),
        },
    },
    "available": {
        "hard": {
            "conversational": (
                "{subject} can only work {scope}",
                "The only times {subject} is available are {scope}",
            ),
            "direct": (
                "Only schedule {subject} {scope}",
                "Do not roster {subject} outside {scope}",
            ),
            "formal": (
                "{subject}'s availability is strictly limited to {scope}",
                "{subject} may be assigned shifts {scope} and at no other time",
            ),
            "terse": (
                "{subject}: available only {scope}",
                "Only workable window for {subject}: {scope}",
            ),
        },
    },
    "preferred_hours": {
        "soft": {
            "conversational": (
                "{subject} would like shifts {scope} when possible",
                "{subject} prefers to work {scope}",
            ),
            "direct": (
                "Favor {scope} when scheduling {subject}",
                "Try to give {subject} shifts {scope}",
            ),
            "formal": (
                "{subject}'s preferred working window is {scope}",
                "Where feasible, assignments for {subject} should fall {scope}",
            ),
            "terse": (
                "{subject}: prefers {scope}",
                "Best shift window for {subject}: {scope}",
            ),
        },
    },
    "min_weekly_hours": {
        "hard": {
            "conversational": (
                "{subject} needs at least {value} hours each week",
                "Make sure {subject} gets {value} or more weekly hours",
            ),
            "direct": (
                "Schedule {subject} for no fewer than {value} hours per week",
                "Give {subject} a minimum of {value} weekly hours",
            ),
            "formal": (
                "{subject}'s weekly total must be at least {value} hours",
                "A weekly minimum of {value} hours applies to {subject}",
            ),
            "terse": (
                "{subject}: min {value} hrs/week",
                "Weekly floor for {subject}: {value} hours",
            ),
        },
        "soft": {
            "conversational": (
                "Ideally, {subject} would get at least {value} hours a week",
                "Try to give {subject} {value} or more weekly hours",
            ),
            "direct": (
                "Aim for a minimum of {value} weekly hours for {subject}",
                "When possible, schedule {subject} at least {value} hours per week",
            ),
            "formal": (
                "The preferred weekly minimum for {subject} is {value} hours",
                "A weekly total of at least {value} hours is requested for {subject}",
            ),
            "terse": (
                "{subject}: target {value}+ hrs/week",
                "Preferred weekly floor for {subject}: {value} hours",
            ),
        },
    },
    "max_weekly_hours": {
        "hard": {
            "conversational": (
                "{subject} can't work more than {value} hours in a week",
                "Keep {subject} at {value} weekly hours or less",
            ),
            "direct": (
                "Do not schedule {subject} beyond {value} hours per week",
                "Cap {subject}'s weekly hours at {value}",
            ),
            "formal": (
                "{subject}'s weekly total must not exceed {value} hours",
                "A maximum of {value} weekly hours applies to {subject}",
            ),
            "terse": (
                "{subject}: max {value} hrs/week",
                "Weekly cap for {subject}: {value} hours",
            ),
        },
        "soft": {
            "conversational": (
                "Try not to give {subject} more than {value} hours a week",
                "Ideally, {subject} would stay under {value} weekly hours",
            ),
            "direct": (
                "Aim to keep {subject} at or below {value} hours per week",
                "Avoid scheduling {subject} over {value} weekly hours when possible",
            ),
            "formal": (
                "The preferred weekly ceiling for {subject} is {value} hours",
                "A weekly load above {value} hours should be avoided for {subject}",
            ),
            "terse": (
                "{subject}: target max {value} hrs/week",
                "Preferred weekly cap for {subject}: {value} hours",
            ),
        },
    },
    "max_consecutive_days": {
        "hard": {
            "conversational": (
                "{subject} can't work more than {value} days in a row",
                "After {value} straight workdays, {subject} needs a day off",
            ),
            "direct": (
                "Limit {subject} to {value} consecutive workdays",
                "Do not roster {subject} for over {value} days straight",
            ),
            "formal": (
                "{subject} may work at most {value} consecutive days",
                "A run of shifts for {subject} must end after {value} days",
            ),
            "terse": (
                "{subject}: max {value} days straight",
                "Consecutive-day cap for {subject}: {value}",
            ),
        },
        "soft": {
            "conversational": (
                "Try to give {subject} a break after {value} days in a row",
                "If possible, don't schedule {subject} over {value} straight days",
            ),
            "direct": (
                "Aim to limit {subject} to {value} consecutive days",
                "Avoid runs longer than {value} days for {subject} when feasible",
            ),
            "formal": (
                "The preferred consecutive-day limit for {subject} is {value}",
                "Runs beyond {value} workdays should be avoided for {subject}",
            ),
            "terse": (
                "{subject}: prefer max {value} days straight",
                "Target consecutive-day cap for {subject}: {value}",
            ),
        },
    },
    "min_rest_hours": {
        "hard": {
            "conversational": (
                "{subject} needs at least {value} hours between shifts",
                "Give {subject} a break of {value} hours or more between shifts",
            ),
            "direct": (
                "Leave at least {value} hours between {subject}'s shifts",
                "Do not give {subject} less than {value} hours of turnaround time",
            ),
            "formal": (
                "{subject} must receive a minimum of {value} rest hours between shifts",
                "A shift gap of at least {value} hours is required for {subject}",
            ),
            "terse": (
                "{subject}: min {value} hrs between shifts",
                "Required turnaround for {subject}: {value} hours",
            ),
        },
        "soft": {
            "conversational": (
                "Try to leave {value} hours between {subject}'s shifts",
                "If possible, give {subject} at least {value} hours of rest",
            ),
            "direct": (
                "Aim for a {value}-hour shift gap for {subject}",
                "Preferably allow {subject} {value} hours between shifts",
            ),
            "formal": (
                "The preferred turnaround time for {subject} is {value} hours",
                "A rest interval of {value} hours is requested for {subject}",
            ),
            "terse": (
                "{subject}: prefer {value} hrs between shifts",
                "Target turnaround for {subject}: {value} hours",
            ),
        },
    },
}

OPENERS = {
    "conversational": ("", "Quick heads-up: ", "Could you note this: ", "For the next schedule, "),
    "direct": ("", "Scheduling request: ", "Please note: ", "For the rota: "),
    "formal": ("", "For scheduling purposes, ", "The following applies: ", "Staffing requirement: "),
    "terse": ("", "Rota note — ", "FYI: ", "Schedule note: "),
}

CONNECTORS = {
    "conversational": (". Also, ", ". And ", "; plus, ", ". One more thing: "),
    "direct": (". Also, ", "; additionally, ", ". Then, ", ". Include this too: "),
    "formal": ("; additionally, ", ". Furthermore, ", ". The request also requires: "),
    "terse": ("; ", ". ", " / ", ", plus "),
}


def _subject(constraint: SchedulingConstraint) -> str:
    return constraint.employee if constraint.employee is not None else "each team member"


def _format_number(value: float | None) -> str:
    if value is None:
        raise ValueError("Numeric constraint is missing its value")
    return str(int(value)) if value.is_integer() else str(value)


def _clock(value: time, clock_style: str) -> str:
    if clock_style == "24h":
        return value.strftime("%H:%M")
    if value.hour == 12 and value.minute == 0 and clock_style == "12lower":
        return "noon"
    hour = value.hour % 12 or 12
    minute = f":{value.minute:02d}" if value.minute else ""
    suffix = "am" if value.hour < 12 else "pm"
    if clock_style == "12upper":
        suffix = suffix.upper()
    return f"{hour}{minute} {suffix}"


def _absolute_date(
    value: date, rng: random.Random, style: str | None = None
) -> tuple[str, str]:
    style = style or rng.choice(("long", "short", "iso", "slash"))
    if style == "long":
        return value.strftime("%B %d, %Y").replace(" 0", " "), "date-long"
    if style == "short":
        return value.strftime("%b %d, %Y").replace(" 0", " "), "date-short"
    if style == "slash":
        return f"{value.month}/{value.day}/{value.year}", "date-slash"
    return value.isoformat(), "date-iso"


def _one_off_date(value: date, reference_date: date, rng: random.Random) -> tuple[str, str]:
    delta = (value - reference_date).days
    if delta == 1 and rng.random() < 0.75:
        return rng.choice(("tomorrow", "the following day", "one day from now")), "date-relative-one"
    if 2 <= delta <= 7 and rng.random() < 0.70:
        return rng.choice(
            (f"in {delta} days", f"this coming {value.strftime('%A')}", f"next {value.strftime('%A')}")
        ), "date-relative-week"
    rendered, render_id = _absolute_date(value, rng)
    return f"on {rendered}", render_id


def _day_phrase(days: list[str], rng: random.Random) -> tuple[str, str]:
    titles = [day.capitalize() for day in days]
    abbreviations = [
        {
            "monday": "Mon",
            "tuesday": "Tue",
            "wednesday": "Wed",
            "thursday": "Thu",
            "friday": "Fri",
            "saturday": "Sat",
            "sunday": "Sun",
        }[day]
        for day in days
    ]
    variant = rng.randrange(4)
    if len(days) == 1:
        options = (
            f"every {titles[0]}",
            f"on {titles[0]}s",
            f"each {titles[0]}",
            f"on {abbreviations[0]}",
        )
    else:
        joined = f"{', '.join(titles[:-1])} and {titles[-1]}"
        options = (f"every {joined}", f"on {joined}", f"each {joined}", " & ".join(abbreviations))
    return options[variant], f"days:{variant}"


def _time_phrase(scope: TimeScope, rng: random.Random) -> tuple[str, str] | None:
    if not scope.start_time and not scope.end_time:
        return None
    clock_style = rng.choices(("12lower", "12upper", "24h"), weights=(5, 2, 3), k=1)[0]
    if scope.start_time and scope.end_time:
        start = _clock(scope.start_time, clock_style)
        end = _clock(scope.end_time, clock_style)
        return rng.choice(
            (
                (f"from {start} to {end}", f"time-range:{clock_style}:from"),
                (f"between {start} and {end}", f"time-range:{clock_style}:between"),
                (f"{start}–{end}", f"time-range:{clock_style}:dash"),
            )
        )
    if scope.start_time:
        start = _clock(scope.start_time, clock_style)
        return rng.choice(
            ((f"after {start}", f"time-after:{clock_style}"), (f"from {start} onward", f"time-onward:{clock_style}"))
        )
    end = _clock(scope.end_time, clock_style)  # type: ignore[arg-type]
    return rng.choice(
        ((f"before {end}", f"time-before:{clock_style}"), (f"no later than {end}", f"time-latest:{clock_style}"))
    )


def _scope(scope: TimeScope, reference_date: date, rng: random.Random) -> tuple[str, list[str]]:
    ids: list[str] = []
    if scope.recurring:
        if not scope.days_of_week:
            raise ValueError("Recurring scope is missing weekdays")
        calendar, calendar_id = _day_phrase(scope.days_of_week, rng)
        ids.append(calendar_id)
        if scope.start_date and scope.end_date:
            date_style = rng.choice(("long", "short", "iso", "slash"))
            start, start_id = _absolute_date(scope.start_date, rng, date_style)
            end, end_id = _absolute_date(scope.end_date, rng, date_style)
            date_variant = rng.randrange(3)
            date_range = (
                f"from {start} through {end}",
                f"between {start} and {end}",
                f"during {start}–{end}",
            )[date_variant]
            calendar = f"{calendar} {date_range}"
            ids.extend((start_id, end_id, f"date-range:{date_variant}"))
    elif scope.start_date:
        calendar, calendar_id = _one_off_date(scope.start_date, reference_date, rng)
        ids.append(calendar_id)
    else:
        raise ValueError("Temporal constraint has no calendar scope")

    rendered_time = _time_phrase(scope, rng)
    if rendered_time is None:
        return calendar, ids
    time_text, time_id = rendered_time
    ids.append(time_id)
    if rng.random() < 0.25:
        return f"{time_text} {calendar}", ids + ["scope:time-first"]
    return f"{calendar} {time_text}", ids + ["scope:calendar-first"]


def _apply_light_noise(text: str, rng: random.Random) -> tuple[str, str | None]:
    if rng.random() >= 0.12:
        return text, None
    candidates = (
        ("hours", "hrs", "noise:hours-shorthand"),
        ("per week", "/week", "noise:per-week-shorthand"),
        ("Please", "Pls", "noise:please-shorthand"),
        ("availability", "availabilty", "noise:availability-typo"),
        ("schedule", "sched", "noise:schedule-shorthand"),
    )
    applicable = [item for item in candidates if item[0] in text]
    if applicable:
        old, new, noise_id = rng.choice(applicable)
        return text.replace(old, new, 1), noise_id
    return text.rstrip("."), "noise:no-final-period"


def _as_continuation(text: str, constraint: SchedulingConstraint) -> str:
    """Lower sentence-initial grammar while preserving employee names."""

    if text.startswith(_subject(constraint)):
        return text
    return text[0].lower() + text[1:]


def render_scenario(scenario: Scenario, rng: random.Random) -> tuple[str, tuple[str, ...], str]:
    style = rng.choices(STYLES, weights=STYLE_WEIGHTS, k=1)[0]
    clauses: list[str] = []
    render_ids: list[str] = [f"style:{style}"]
    for constraint in scenario.output.constraints:
        variants = CLAUSE_TEMPLATES[constraint.type][constraint.priority][style]
        variant_index = rng.randrange(len(variants))
        scope_text = ""
        if constraint.time_scope is not None:
            scope_text, scope_ids = _scope(constraint.time_scope, scenario.reference_date, rng)
            render_ids.extend(scope_ids)
        clauses.append(
            variants[variant_index].format(
                subject=_subject(constraint),
                scope=scope_text,
                value=_format_number(constraint.value) if constraint.value is not None else "",
            )
        )
        render_ids.append(f"clause:{constraint.type}:{constraint.priority}:{style}:{variant_index}")

    opener = rng.choice(OPENERS[style])
    if opener.startswith("Please") and clauses[0].startswith("Please"):
        opener = ""
    first_clause = _as_continuation(clauses[0], scenario.output.constraints[0]) if opener else clauses[0]
    text = opener + first_clause
    for index, clause in enumerate(clauses[1:], 1):
        connector_index = rng.randrange(len(CONNECTORS[style]))
        connector = CONNECTORS[style][connector_index]
        if connector != ". ":
            clause = _as_continuation(clause, scenario.output.constraints[index])
        text += connector + clause
        render_ids.append(f"connector:{style}:{connector_index}")
    text = text.rstrip(".") + "."
    text, noise_id = _apply_light_noise(text, rng)
    if noise_id:
        render_ids.append(noise_id)
    return text, tuple(render_ids), style

"""Decision tasks.

A task is a label space, the wording every backend sees, and labelled examples.
Loaders are lazy so that a task needing a download cannot break an offline run.
"""

from __future__ import annotations

from ..types import TaskSpec

_TASKS = {
    "synthetic": "rule-generated support triage, 4 labels, no download",
    "banking77": "banking intent classification, 77 labels (needs datasets)",
}


def list_tasks() -> dict[str, str]:
    return dict(_TASKS)


def load_task(name: str, **kwargs) -> TaskSpec:
    if name == "synthetic":
        from .synthetic import load

        return load(**kwargs)
    if name == "banking77":
        from .banking77 import load

        return load(**kwargs)
    raise ValueError(f"unknown task: {name}. Known: {', '.join(_TASKS)}")

"""A rule-generated support-triage task.

Deliberately offline and deterministic: it is the task the test suite and CI
run against, so the whole pipeline can be exercised with no download, no API key
and no model. Because it is generated from templates it is also the one task
where the true decision boundary is known, which makes it useful for catching a
backend that is confidently wrong rather than merely inaccurate.
"""

from __future__ import annotations

import random

from ..types import Example, TaskSpec

CRITERIA = {
    "billing": "Payment, invoice, refund, subscription or charge problems",
    "technical": "Bugs, errors, crashes, integration or API problems",
    "account": "Login, password, access, or profile and settings changes",
    "sales": "Pricing questions, plan comparisons, quotes or upgrades",
}

_TEMPLATES = {
    "billing": [
        "I was charged {n} times for the same order this month",
        "Please refund invoice {inv}, the amount is wrong",
        "My subscription renewed but the card was declined",
        "Why is my invoice {inv} higher than last month",
        "Cancel my plan and refund the last payment please",
    ],
    "technical": [
        "The API returns a {code} error on every upload",
        "The dashboard crashes when I open the reports tab",
        "Webhooks stopped firing after the {code} update",
        "Getting a timeout when syncing more than {n} records",
        "The export button does nothing in Firefox",
    ],
    "account": [
        "I cannot log in, the reset email never arrives",
        "Please change the email address on my account",
        "Two factor authentication is locking me out",
        "How do I add {n} more users to my workspace",
        "I need to transfer ownership of the account",
    ],
    "sales": [
        "What is the price difference between pro and enterprise",
        "Can I get a quote for {n} seats",
        "Do you offer a discount for annual billing",
        "Which plan includes the audit log",
        "We are comparing you against a competitor, can you help",
    ],
}


def load(n: int = 200, seed: int = 0) -> TaskSpec:
    rng = random.Random(seed)
    labels = list(CRITERIA)
    examples: list[Example] = []
    for i in range(n):
        label = labels[i % len(labels)]
        template = rng.choice(_TEMPLATES[label])
        text = template.format(
            n=rng.randint(2, 50),
            inv=f"INV-{rng.randint(1000, 9999)}",
            code=rng.choice(["500", "403", "429", "v2.1"]),
        )
        examples.append(Example(uid=f"syn-{i:04d}", text=text, gold=label))
    rng.shuffle(examples)
    return TaskSpec(
        name="synthetic",
        instructions="Which team should handle this support ticket",
        criteria=CRITERIA,
        examples=examples,
        source="generated offline from templates; no network required",
    )

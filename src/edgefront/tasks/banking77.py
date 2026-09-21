"""Banking77 - 77-way banking intent classification.

Chosen because it is public, genuinely hard (77 fine-grained labels), and
already used by at least one published Jev calibration study, so results here
can be cross-checked against someone else's rather than standing alone.

Needs `datasets`; every other part of edgefront runs without it.
"""

from __future__ import annotations

from ..types import Example, TaskSpec

# Label names carry the meaning here, so criteria are derived from them rather
# than hand-written: 77 hand-written descriptions would be 77 chances to bias
# one backend against another.
_HINT = "Customer's banking intent: {}"


def _humanise(label: str) -> str:
    return label.replace("_", " ").strip()


_SOURCE_REPO = "legacy-datasets/banking77"


def load(n: int = 500, split: str = "test", seed: int = 0) -> TaskSpec:
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "banking77 needs the `datasets` package: pip install datasets"
        ) from exc

    # PolyAI/banking77, the original upload, ships a python loading script.
    # `datasets` >= 3.0 refuses to execute those for security reasons and
    # raises "Dataset scripts are no longer supported" - not a transient
    # failure, every load of that repo id now fails this way. HF's own
    # migration of the identical rows to parquet lives at
    # legacy-datasets/banking77; same text/label schema, no script involved.
    ds = load_dataset(_SOURCE_REPO, split=split)
    names = ds.features["label"].names
    ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))

    criteria = {name: _HINT.format(_humanise(name)) for name in names}
    examples = [
        Example(uid=f"b77-{i:04d}", text=row["text"], gold=names[row["label"]])
        for i, row in enumerate(ds)
    ]
    return TaskSpec(
        name="banking77",
        instructions="What is the customer asking about",
        criteria=criteria,
        examples=examples,
        source=f"{_SOURCE_REPO} {split} split, {len(examples)} examples, seed {seed}",
    )

# edgefront — state of the project

Working notes for whoever picks this up next. Written 2026-09-22.

## What this is

A benchmark that answers one question: *for my decision task, do I need a hosted
typed-decision model, or does a small local model match it?* It runs the same
examples, label space and wording through every backend and reports accuracy,
calibration error, the latency distribution and cost per million calls, then
prints a verdict.

Repo: https://github.com/shivpratapsinghpanwar/edgefront · MIT · CI green on 3.10 + 3.12.

## What already works

- `bench` / `report` / `verify` / `tasks` CLI, data-doctor house style, exit code
  is the verdict.
- Backends: `rules` (keyword floor), `stub` (deterministic, CI), `hf[:model-id]`
  (torch zero-shot NLI), `onnx:<file>:<tokenizer>[:precision]`, `jev` (hosted).
- `quantize/`: HF → ONNX export and dynamic INT8. Measured on
  `typeform/distilbert-base-uncased-mnli`: **267.9 MB → 67.3 MB, 4.0× smaller**.
- 21 tests, all offline — no API key, no model, no network.

## The measured result so far

60 examples of the bundled `synthetic` task, Windows CPU, from India:

| backend | acc | ECE | p50 ms | p99 ms | $/1M | offline |
|---|---:|---:|---:|---:|---:|:--:|
| rules | 0.450 | 0.099 | 0.013 | 0.368 | 0.0003 | yes |
| onnx fp32 | 0.383 | 0.165 | 481 | 3911 | 12.9 | yes |
| onnx int8 | 0.500 | 0.284 | 249 | 523 | 6.69 | yes |
| jev | 0.950 | 0.036 | 386 | 557 | 3.59 | no |

**The hosted model wins decisively** — 95% vs 50%, better calibrated, and
cheaper, because the local NLI approach scores one hypothesis per label and
burns 4× the passes. This is the opposite of the project's original hypothesis.
Keep it that way in the README: the finding is the product.

Three things worth carrying into the writeup:

1. **The vendor's published 70–500 ms does not hold from this network position.**
   Measured p50 386 ms, p99 557 ms, and p99 1254 ms on an earlier run. This is
   why `environment()` records host and UTC time in every result document.
2. **INT8 beat FP32 on accuracy** (0.500 vs 0.383) and was ~2× faster. On 60
   examples the accuracy difference is noise; the latency and size are real.
3. **Local cost scales with label count**, since it is one forward pass per
   label. At banking77's 77 labels it gets much worse, not better.

## Gotcha that cost an hour — do not regress it

`typeform/distilbert-base-uncased-mnli` orders its labels
**ENTAILMENT, NEUTRAL, CONTRADICTION** — entailment is index **0**, not the
conventional 2. The ONNX backend originally assumed `n_classes - 1` and was
scoring the *contradiction* logit, i.e. the opposite of the question. Accuracy
read 0.167 instead of 0.417 and looked plausible.

`_entailment_index()` in `backends/onnx_local.py` now reads it from the
checkpoint config. **There is no regression test for this yet — add one.** It is
the single most dangerous class of bug in this project: silently wrong rather
than loudly wrong.

## Next, in order

1. **Regression test for the entailment index.** Assert that for a checkpoint
   whose `id2label` puts entailment at 0, the resolved index is 0. Mock the
   config rather than downloading a model, so it stays offline.
2. **Commit and push the current working tree** — the ONNX backend, `quantize/`,
   the provider-default fix and the entailment fix are all uncommitted.
3. **Update the README with the real table above**, replacing the placeholder
   numbers. State plainly that the hosted model won and why.
4. **The Kaggle job** (see below).
5. `measure/calibration.py` reliability curve already exists in the result JSON
   under `meta.reliability_curve` — render it in `report.py`.

## The Kaggle job

banking77 is 77 labels, so the local NLI backend needs 77 forward passes per
example: roughly **19 s per example on this laptop's CPU**, which is unusable.
A T4 makes it tractable.

- Kaggle auth already works here (`python -m kaggle kernels list --user
  shivpratap0007` succeeds; the refresh token is valid).
- There is a battle-tested runner to model it on at
  `C:\Users\hp\Synthetic_Data_Factory\kaggle_runner\` — preflight checks,
  push, poll with backoff, collect, and a compact machine-readable `result.json`.
  Treat it as a pattern, do not copy employer code.
- The job should: pip install edgefront from the public GitHub repo, load
  banking77 via `datasets`, run `hf` on CUDA plus `onnx` int8 on CPU, and write
  the result document to `/kaggle/working`.
- **Do not put the API key in a Kaggle kernel.** Run the hosted arm locally and
  merge the two result documents, or use a Kaggle Secret.

## Constraints that are not negotiable

- **The owner is employed.** No employer code, data or hardware in this repo.
  The `Synthetic_Data_Factory` and `surveillance_use_case` runners are
  conceptual precedent only — reimplement clean-room.
- **Commits are authored by the repo owner only.** No AI attribution trailers,
  no `Co-Authored-By`. This was asked for explicitly.
- **Honest reporting.** If a number is unflattering, publish it. The one thing
  that would kill this project is a benchmark caught flattering its own premise.
- The API key lives at `C:\Users\hp\Resumes\api_type.txt`, format `apikey_…`.
  Never commit it; never echo it.

## Known weaknesses to be honest about in the README

- The bundled `synthetic` task is generated from templates that share vocabulary
  with the label descriptions, so a keyword baseline does unusually well on it.
  It exists to exercise the pipeline offline in CI, not to rank models. Say so.
- The local contender is zero-shot NLI, chosen because the hosted model is also
  zero-shot. A task-specific fine-tune would score far higher — and that is a
  legitimate objection to answer in the README, not to hide.
- Question wording is an experimental variable. TypeSafe's own guidance says
  agents write poor questions and expect to refine them collaboratively; bad
  `criteria` wording would unfairly penalise the hosted model.

# edgefront

Do you need a hosted decision model, or does a small local model match it?
Measure it on your own task instead of guessing.

```bash
pip install edgefront
edgefront bench --task synthetic --backends rules,stub
```

## Why

Hosted "typed decision" models return a label plus calibrated probabilities in
70–500 ms for a fraction of an LLM's price. A quantized classifier on your own
hardware returns the same shape of answer in single-digit milliseconds for
nothing, and works offline.

Which one you should deploy depends on your task, your latency budget and your
volume. This measures all three and prints a verdict.

## What it does

Runs the same examples, the same label space and the same wording through every
backend, then reports accuracy, calibration error, the latency distribution and
cost per million calls.

```
backend                 acc    ECE    p50ms    p99ms      $/1M  offline
-----------------------------------------------------------------------
rules                 0.450  0.099    0.013    0.103  3.00e-04  yes
distilbert-mnli-fp32  0.383  0.165      322      459      8.66   yes

verdict: rules leads at 45.0% accuracy
         the keyword baseline is within 0.0 pts of the best backend -
         this task may be too easy to separate them
```

That output is real, and it is a good example of the tool doing its job: on the
bundled synthetic task a keyword baseline beats a 67M-parameter zero-shot model,
and the report says so rather than quietly reporting the model's number alone.
It also means **the synthetic task is not a good discriminator** — it exists so
the pipeline can be exercised offline in CI, not to prove anything about models.
Use `banking77` or your own data for a real answer.

## Design rules

These are the parts that decide whether a benchmark is worth reading:

- **A keyword baseline runs by default.** If a frontier model is only a few
  points above keyword matching, that is the most important number on the page,
  and a benchmark without a floor hides it.
- **Latency is a distribution, never a mean.** p50/p90/p99/max. A mean hides the
  tail, and the tail is what breaks a real-time budget.
- **The first 3 calls per backend are discarded** as warmup.
- **Backends run sequentially** so they cannot contend for CPU and corrupt each
  other's timings.
- **A failed call counts as wrong, not as missing.** A backend that errors on
  10% of inputs is not accurate on the rest.
- **Cost assumptions are printed next to the cost.** Local cost is amortised
  hardware plus power, which is arguable, so the arithmetic is shown.
- **The comparison is zero-shot on both sides.** A hosted model has not seen
  your labels, so the local contender is scored zero-shot too via NLI
  entailment. A fine-tuned classifier would win while answering a different
  question.

## Usage

```bash
edgefront tasks                     # list decision tasks
edgefront bench --task synthetic --backends rules,stub --json out.json
edgefront report out.json --out BENCH.md
edgefront verify out.json --min-acc 0.85 --max-p99 50
```

`verify` exits 1 when a threshold is missed, so it drops straight into CI.

### Backends

| token | what it is | needs |
|---|---|---|
| `rules` | keyword overlap baseline | nothing |
| `stub` | deterministic fake, for tests | nothing |
| `hf[:model-id]` | local zero-shot NLI classifier | `edgefront[local]` |
| `jev` | TypeSafe Jev, hosted | `edgefront[jev]` + `TYPESAFE_API_KEY` |

Adding a backend means implementing two methods — `predict()` and `meta()`.
See `src/edgefront/backends/stub.py`; it is the whole contract.

### Install

```bash
pip install edgefront            # core: stdlib only, runs bench/report/verify
pip install 'edgefront[local]'   # local models
pip install 'edgefront[jev]'     # hosted backend
pip install 'edgefront[all]'
```

The core install has no dependencies, so `report` and `verify` run anywhere,
including in CI with no model and no network.

## Status

Early. The synthetic task and the rules, stub and local backends work; the
hosted backend is implemented but the numbers in this README were produced
without one, so no hosted-vs-local verdict has been published yet. The
quantization axis (fp32 → int8 → int4) is next.

## License

MIT

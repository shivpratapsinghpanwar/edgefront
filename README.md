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

## Results

60 examples of the bundled `synthetic` task, Windows CPU, run from India against
the hosted endpoint:

| backend | acc | ECE | p50 ms | p99 ms | $/1M | offline |
|---|---:|---:|---:|---:|---:|:--:|
| rules | 0.450 | 0.099 | 0.013 | 0.368 | 0.0003 | yes |
| onnx fp32 | 0.383 | 0.165 | 481 | 3911 | 12.9 | yes |
| onnx int8 | 0.500 | 0.284 | 249 | 523 | 6.69 | yes |
| jev (hosted) | 0.950 | 0.036 | 386 | 557 | 3.59 | no |

**The hosted model wins, decisively — by 45 points — and it is also better
calibrated and cheaper.** This is the opposite of the project's original
hypothesis, and it is the finding, not an embarrassment to bury: the local
zero-shot NLI approach scores one entailment hypothesis per label, so it pays
for a full forward pass *per label, per example*. On this 4-label task that is
already 4x the inference cost of one hosted call; the hosted model answers all
labels in a single request. Quantizing the local model to INT8 clawed back
some of that (see below) but did not close a 45-point accuracy gap, and paying
per-forward-pass cost on a worse model is not a trade worth making.

Three things worth carrying forward honestly:

1. **The vendor's published 70-500 ms latency does not hold from this network
   position.** Measured p50 386 ms, p99 557 ms (p99 hit 1254 ms on an earlier
   run). Hosted latency is a function of where you call it from, not just the
   model — `environment()` records host and UTC time in every result document
   so this doesn't get generalized past the network it was measured on.
2. **INT8 beat FP32 on both speed and size, and matched or beat it on
   accuracy** (0.500 vs 0.383 — on 60 examples that gap is noise, but the ~2x
   latency improvement and the 4.0x smaller checkpoint are real and repeatable:
   `typeform/distilbert-base-uncased-mnli` went from 267.9 MB to 67.3 MB).
   There was no accuracy tax for quantizing dynamically, at least here.
3. **Local cost scales with label count, and it scales badly.** The local
   backend is one forward pass per label; the hosted backend is not. At
   banking77's 77 labels the local backend gets dramatically slower and more
   expensive, not better — see the Kaggle job below, built specifically because
   77 labels made this untenable on a laptop CPU (roughly 19 s/example).

This output is real, and it is a good example of the tool doing its job: it
does not quietly report the model's number alone, or let a locally-run
benchmark flatter local hardware. See `BENCH.md` for the full report with
cost-basis disclosures, and `bench_full.json` for the raw result document.

The bundled `synthetic` task above is generated from templates that share
vocabulary with the label descriptions, so a keyword baseline does unusually
well on it — that's a known weakness, not a discriminator; see below. It
exists to exercise the pipeline offline in CI, not to prove anything about
models. Use `banking77` or your own data for a real answer.

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

## Known weaknesses

Said plainly, because a benchmark that hides its own weak points is worse than
no benchmark:

- **The bundled `synthetic` task is generated from templates that share
  vocabulary with the label descriptions**, so a keyword baseline does
  unusually well on it (0.450 accuracy above, competitive with the local
  models). It exists to exercise the pipeline offline in CI, not to rank
  models — it is not a good discriminator, and the numbers above should not be
  read as "keyword matching nearly beats a neural model in general."
- **The local contender is zero-shot NLI, not a fine-tuned classifier**, chosen
  because the hosted model is also zero-shot and the comparison has to hold
  that variable constant. A task-specific fine-tune of the same small model
  would very likely score far higher than either backend above. That is a
  legitimate objection to this benchmark's framing, not one to hide: if you can
  afford to fine-tune on your own labels, do that comparison instead — this
  tool answers "zero-shot local vs. zero-shot hosted," not "the best local
  model you could build vs. hosted."
- **Question wording is an experimental variable, not a constant.** The
  hypothesis template (`"This text is about {}."`) and the task's `criteria`
  wording both affect both backends, but not necessarily equally — TypeSafe's
  own guidance says agents write poor questions on the first pass and expect to
  refine them collaboratively. Bad wording here would unfairly penalize the
  hosted model, which is one more reason to rerun this on your own task and
  wording rather than trust the numbers above at face value.

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
| `hf[:model-id]` | local zero-shot NLI classifier, torch | `edgefront[local]` |
| `onnx:<file>:<tokenizer>[:precision]` | local zero-shot NLI, ONNX Runtime (fp32/int8) | `edgefront[local]` |
| `jev` | TypeSafe Jev, hosted | `edgefront[jev]` + `TYPESAFE_API_KEY` |

`edgefront.quantize` exports an HF checkpoint to ONNX and quantizes it to
dynamic INT8 (`export_onnx`, `quantize_int8`) so you can produce the
`onnx:model.onnx:tokenizer-id` and `onnx:model.int8.onnx:tokenizer-id:int8`
backends above from any HF sequence-classification checkpoint.

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

Early, but the core question has a published answer now: see Results above —
on the bundled task, hosted beats local zero-shot NLI by 45 points, at a lower
cost per call. The `rules`, `stub`, `hf`, `onnx` (fp32 and int8) and `jev`
backends all work; export/quantization from an HF checkpoint to ONNX INT8 is
done (`edgefront.quantize`). Next: a real-task run on banking77 (77 labels)
comparing torch-on-GPU against onnx-int8-on-CPU, and rendering the calibration
reliability curve that's already in the result JSON (`meta.reliability_curve`)
into the markdown report.

## License

MIT

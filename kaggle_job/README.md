# Kaggle job: banking77 on a GPU

banking77 has 77 labels. The local zero-shot NLI backend scores one entailment
hypothesis per label per example, so 77 labels costs roughly **19 s/example on
a laptop CPU** — untenable for a few hundred examples. This kernel runs it on
a Kaggle GPU instead, where it takes minutes.

It does **not** call the hosted (`jev`) backend — there is no API key on this
kernel, on purpose. The two arms are combined afterwards on your own machine.

## What it does

1. `pip install`s edgefront from the public GitHub repo (`main`), so it always
   benchmarks whatever is currently pushed.
2. Loads `banking77` (300 examples by default) via the `datasets` package.
3. Runs `rules` (baseline), `distilbert-mnli` on torch/CUDA, and the same
   checkpoint exported to ONNX and quantized to INT8, run on CPU.
4. Writes `/kaggle/working/banking77_local.json` — a normal edgefront result
   document, just missing the hosted backend.

## Push and run

```bash
cd kaggle_job
kaggle kernels push
```

Then watch it from the Kaggle UI or:

```bash
kaggle kernels status shivpratap0007/edgefront-banking77
kaggle kernels output shivpratap0007/edgefront-banking77 -p ./output
```

`./output/banking77_local.json` is the result document.

## Combine with the hosted arm

Run the hosted backend locally, where the API key actually lives:

```bash
export TYPESAFE_API_KEY=...   # never on the Kaggle kernel
edgefront bench --task banking77 --backends jev --limit 300 --json banking77_jev.json
```

Then merge the two into one verdict:

```bash
edgefront merge output/banking77_local.json banking77_jev.json \
  --json banking77_full.json --md BANKING77.md
```

`merge` refuses to combine documents that ran a different number of examples,
or that both contain the same backend name — either would make "who won by
how much" meaningless rather than just imprecise.

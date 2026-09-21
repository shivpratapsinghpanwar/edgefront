"""Export a HuggingFace sequence classifier to ONNX, then quantize it.

Kept deliberately thin. The point of this project is to *measure* what each
precision costs you, so the export path has to be the ordinary one a
practitioner would reach for - not a hand-tuned graph that flatters the result.
"""

from __future__ import annotations

from pathlib import Path

OPSET = 17


def export_onnx(
    model_name: str,
    out_dir: str | Path,
    *,
    max_length: int = 256,
    overwrite: bool = False,
) -> Path:
    """Export ``model_name`` to ``out_dir/model.onnx`` and return the path."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / "model.onnx"
    if target.exists() and not overwrite:
        return target

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()

    encoded = tokenizer(
        ["premise text"],
        ["hypothesis text"],
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding="max_length",
    )
    inputs = (encoded["input_ids"], encoded["attention_mask"])

    with torch.inference_mode():
        torch.onnx.export(
            model,
            inputs,
            str(target),
            input_names=["input_ids", "attention_mask"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "sequence"},
                "attention_mask": {0: "batch", 1: "sequence"},
                "logits": {0: "batch"},
            },
            opset_version=OPSET,
            do_constant_folding=True,
        )
    return target


def quantize_int8(src: str | Path, dst: str | Path | None = None) -> Path:
    """Dynamic INT8 quantization - the default a practitioner would try first.

    Dynamic rather than static because it needs no calibration set, which keeps
    the comparison honest: a statically quantized model tuned on this task's
    data would be a different, easier experiment.
    """
    from onnxruntime.quantization import QuantType, quantize_dynamic

    src = Path(src)
    dst = Path(dst) if dst else src.with_name("model.int8.onnx")
    if dst.exists():
        return dst
    quantize_dynamic(
        model_input=str(src),
        model_output=str(dst),
        weight_type=QuantType.QInt8,
    )
    return dst


def size_mb(path: str | Path) -> float:
    return round(Path(path).stat().st_size / 1e6, 2)

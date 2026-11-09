"""Dependency-light quantization issue analysis for arbitrary PyTorch models.

This module intentionally knows nothing about Linger. It observes standard
``torch.nn.Module`` parameters and forward values, so it can be used before,
during, or after a QAT experiment without changing the training code.
"""

from __future__ import annotations

import csv
import fnmatch
import html
import importlib
import importlib.util
import json
import math
import random
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .diagnostics import ANOMALY_RULES, classify_anomaly


class AnalysisError(RuntimeError):
    """Raised when the standalone analyzer cannot satisfy its input contract."""


def _torch():
    try:
        import torch
        import torch.nn as nn
    except ImportError as exc:  # pragma: no cover - depends on user environment
        raise AnalysisError(
            "PyTorch is required. Install the project's requirements before running "
            "the analyzer: python -m pip install -r requirements.txt"
        ) from exc
    return torch, nn


@dataclass
class AnalysisConfig:
    output_dir: str = "quant_analysis_report"
    max_batches: int = 32
    max_samples: int = 100_000
    histogram_bins: int = 64
    max_histograms: int = 12
    device: str = "auto"
    include: List[str] = field(default_factory=list)
    exclude: List[str] = field(default_factory=list)
    seed: int = 1234
    strict: bool = True
    model_label: str = "quantized"
    float_label: str = "float"


def _jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return str(value)


def _import_callable(spec: str) -> Callable[..., Any]:
    if ":" not in spec:
        raise AnalysisError("--model-factory must use MODULE:CALLABLE or FILE.py:CALLABLE")
    module_spec, attr_name = spec.rsplit(":", 1)
    if Path(module_spec).is_file():
        module_path = Path(module_spec).resolve()
        name = f"quant_analysis_model_{abs(hash(str(module_path)))}"
        module_spec_obj = importlib.util.spec_from_file_location(name, module_path)
        if module_spec_obj is None or module_spec_obj.loader is None:
            raise AnalysisError(f"Cannot import model module: {module_path}")
        module = importlib.util.module_from_spec(module_spec_obj)
        module_spec_obj.loader.exec_module(module)
    else:
        try:
            module = importlib.import_module(module_spec)
        except ImportError as exc:
            raise AnalysisError(f"Cannot import model module '{module_spec}': {exc}") from exc
    try:
        factory = getattr(module, attr_name)
    except AttributeError as exc:
        raise AnalysisError(f"Model factory '{attr_name}' not found in '{module_spec}'") from exc
    if not callable(factory):
        raise AnalysisError(f"Model factory is not callable: {spec}")
    return factory


def _load_checkpoint(model: Any, path: Optional[str], strict: bool) -> Dict[str, Any]:
    if not path:
        return {"path": None, "missing_keys": [], "unexpected_keys": []}
    torch, _ = _torch()
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise AnalysisError(f"Checkpoint does not exist: {checkpoint_path}")
    try:
        payload = torch.load(str(checkpoint_path), map_location="cpu")
    except Exception as exc:
        raise AnalysisError(f"Failed to load checkpoint '{checkpoint_path}': {exc}") from exc
    if hasattr(payload, "state_dict") and callable(payload.state_dict):
        payload = payload.state_dict()
    if not isinstance(payload, Mapping):
        raise AnalysisError(
            f"Checkpoint '{checkpoint_path}' must be a state_dict or a mapping containing one"
        )
    state = payload
    for key in ("state_dict", "model_state_dict", "model", "weights"):
        candidate = payload.get(key)
        if isinstance(candidate, Mapping):
            state = candidate
            break
    try:
        result = model.load_state_dict(state, strict=strict)
    except RuntimeError as exc:
        hint = " Try --non-strict only when the missing keys are expected." if strict else ""
        raise AnalysisError(f"Checkpoint keys do not match the model: {exc}.{hint}") from exc
    return {
        "path": str(checkpoint_path),
        "missing_keys": list(getattr(result, "missing_keys", [])),
        "unexpected_keys": list(getattr(result, "unexpected_keys", [])),
    }


def _device(requested: str) -> Any:
    torch, _ = _torch()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        selected = torch.device(requested)
    except Exception as exc:
        raise AnalysisError(f"Invalid device '{requested}': {exc}") from exc
    if selected.type == "cuda" and not torch.cuda.is_available():
        raise AnalysisError("CUDA was requested but is not available")
    return selected


def _matches(name: str, patterns: Sequence[str]) -> bool:
    return not patterns or any(fnmatch.fnmatch(name, pattern) for pattern in patterns)


def _move(value: Any, device: Any) -> Any:
    torch, _ = _torch()
    if isinstance(value, Mapping):
        return {key: _move(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_move(item, device) for item in value)
    if isinstance(value, list):
        return [_move(item, device) for item in value]
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=True)
    return value


def _as_tensor(value: Any) -> Any:
    torch, _ = _torch()
    if value is None:
        return None
    dequantize = getattr(value, "dequantize", None)
    if callable(dequantize):
        try:
            value = dequantize()
        except Exception:
            pass
    if not isinstance(value, torch.Tensor):
        try:
            value = torch.as_tensor(value)
        except Exception:
            return None
    if not (value.is_floating_point() or value.is_complex()):
        value = value.float()
    else:
        value = value.float()
    return value.detach()


class _Stats:
    def __init__(self, max_samples: int, seed: int) -> None:
        self.max_samples = max(1, int(max_samples))
        self._rng = random.Random(seed)
        self.count = 0
        self.finite_count = 0
        self.nan_count = 0
        self.inf_count = 0
        self.zero_count = 0
        self.minimum = math.inf
        self.maximum = -math.inf
        self.abs_max = 0.0
        self._sum = 0.0
        self._sum_sq = 0.0
        self.samples: List[float] = []

    def add(self, value: Any) -> None:
        torch, _ = _torch()
        tensor = _as_tensor(value)
        if tensor is None or tensor.numel() == 0:
            return
        flat = tensor.reshape(-1)
        self.count += int(flat.numel())
        finite_mask = torch.isfinite(flat)
        finite = flat[finite_mask]
        self.nan_count += int(torch.isnan(flat).sum().item())
        self.inf_count += int(torch.isinf(flat).sum().item())
        if finite.numel() == 0:
            return
        finite_cpu = finite.detach().to("cpu", dtype=torch.float64)
        self.finite_count += int(finite_cpu.numel())
        self.zero_count += int((finite_cpu == 0).sum().item())
        self.minimum = min(self.minimum, float(finite_cpu.min().item()))
        self.maximum = max(self.maximum, float(finite_cpu.max().item()))
        self.abs_max = max(self.abs_max, float(finite_cpu.abs().max().item()))
        self._sum += float(finite_cpu.sum().item())
        self._sum_sq += float((finite_cpu * finite_cpu).sum().item())

        # Systematic sampling keeps hook overhead bounded for large activations.
        sample_limit = min(self.max_samples, 4096)
        stride = max(1, int(math.ceil(finite_cpu.numel() / sample_limit)))
        candidates = finite_cpu[::stride].tolist()
        for item in candidates:
            if len(self.samples) < self.max_samples:
                self.samples.append(float(item))
            else:
                index = self._rng.randrange(self.finite_count)
                if index < self.max_samples:
                    self.samples[index] = float(item)

    def summary(self, bins: int) -> Dict[str, Any]:
        samples = sorted(self.samples)
        if self.finite_count:
            mean = self._sum / self.finite_count
            variance = max(0.0, self._sum_sq / self.finite_count - mean * mean)
            std = math.sqrt(variance)
            percentiles = {
                f"p{p}": _percentile(samples, p / 100.0) for p in (50, 90, 99, 99.9)
            }
        else:
            mean = std = None
            percentiles = {f"p{p}": None for p in (50, 90, 99, 99.9)}
        histogram = _histogram(samples, bins) if samples else {"edges": [], "counts": []}
        return {
            "count": self.count,
            "finite_count": self.finite_count,
            "nan_count": self.nan_count,
            "inf_count": self.inf_count,
            "zero_ratio": self.zero_count / self.finite_count if self.finite_count else None,
            "min": None if self.minimum == math.inf else self.minimum,
            "max": None if self.maximum == -math.inf else self.maximum,
            "abs_max": self.abs_max if self.finite_count else None,
            "mean": mean,
            "std": std,
            **percentiles,
            "histogram": histogram,
        }


def _percentile(values: Sequence[float], fraction: float) -> Optional[float]:
    if not values:
        return None
    if len(values) == 1:
        return float(values[0])
    position = (len(values) - 1) * fraction
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(values[lower])
    weight = position - lower
    return float(values[lower] * (1.0 - weight) + values[upper] * weight)


def _histogram(values: Sequence[float], bins: int) -> Dict[str, Any]:
    if not values:
        return {"edges": [], "counts": []}
    low, high = float(min(values)), float(max(values))
    if low == high:
        return {"edges": [low, high], "counts": [len(values)]}
    bins = max(4, int(bins))
    width = (high - low) / bins
    counts = [0] * bins
    for value in values:
        index = int((value - low) / width)
        counts[min(bins - 1, max(0, index))] += 1
    edges = [low + width * index for index in range(bins + 1)]
    return {"edges": edges, "counts": counts}


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")[:100] or "layer"


def _metadata(module: Any) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    names = ("scale", "threshold", "learning_data", "running_data", "clamp_value", "data_bits", "bits")
    for owner_name, owner in [("module", module)] + list(module.named_modules()):
        for name in names:
            if not hasattr(owner, name):
                continue
            try:
                value = getattr(owner, name)
                if hasattr(value, "detach"):
                    value = value.detach().cpu().reshape(-1)
                    if value.numel() == 1:
                        value = value.item()
                    elif value.numel() <= 16:
                        value = value.tolist()
                    else:
                        continue
                if isinstance(value, (int, float, str, bool)):
                    result[f"{owner_name}.{name}"] = _jsonable(value)
            except Exception:
                continue
    return result


class _ModelObserver:
    def __init__(self, model: Any, config: AnalysisConfig, label: str) -> None:
        torch, nn = _torch()
        self.torch = torch
        self.model = model
        self.config = config
        self.label = label
        self.stats: Dict[Tuple[str, str], _Stats] = {}
        self.modules: Dict[str, Any] = {}
        self.times: Dict[str, List[float]] = {}
        self.handles: List[Any] = []
        self.output_samples: List[Any] = []
        for name, module in model.named_modules():
            if not name or not _matches(name, config.include) or any(
                fnmatch.fnmatch(name, pattern) for pattern in config.exclude
            ):
                continue
            self.modules[name] = module
            for parameter_name, parameter in module.named_parameters(recurse=False):
                self._add(name, f"weight" if parameter_name == "weight" else parameter_name, parameter, module)
            is_leaf = not any(module.children())
            if is_leaf:
                self.handles.append(module.register_forward_pre_hook(self._pre_hook(name)))
                self.handles.append(module.register_forward_hook(self._post_hook(name)))

    def _add(self, module_name: str, role: str, value: Any, module: Any) -> None:
        key = (module_name, role)
        if key not in self.stats:
            self.stats[key] = _Stats(self.config.max_samples, self.config.seed + len(self.stats))
        self.stats[key].add(value)
        self._metadata_for[key] = _metadata(module)

    @property
    def _metadata_for(self) -> Dict[Tuple[str, str], Dict[str, Any]]:
        if not hasattr(self, "__metadata"):
            self.__metadata = {}
        return self.__metadata

    def _pre_hook(self, name: str) -> Callable[..., Any]:
        def hook(_module: Any, _inputs: Tuple[Any, ...]) -> None:
            self.times.setdefault(name, []).append(-time.perf_counter())
        return hook

    def _post_hook(self, name: str) -> Callable[..., Any]:
        def hook(module: Any, inputs: Tuple[Any, ...], output: Any) -> None:
            timings = self.times.setdefault(name, [])
            if timings and timings[-1] < 0:
                timings[-1] += time.perf_counter()
                timings[-1] *= 1000.0
            for index, value in enumerate(inputs):
                tensor = _as_tensor(value)
                if tensor is not None:
                    self._add(name, "activation_input" if index == 0 else f"activation_input_{index}", tensor, module)
            values = output if isinstance(output, (tuple, list)) else (output,)
            for index, value in enumerate(values):
                tensor = _as_tensor(value)
                if tensor is not None:
                    self._add(name, "activation_output" if index == 0 else f"activation_output_{index}", tensor, module)
        return hook

    def observe(self, inputs: Any, max_batches: int) -> Any:
        self.model.eval()
        with self.torch.inference_mode():
            for index, batch in enumerate(_batches(inputs)):
                if index >= max_batches:
                    break
                output = _call_model(self.model, batch)
                sample = _first_tensor(output)
                if sample is not None:
                    self.output_samples.append(sample[:100_000].to("cpu"))
        return _combine_output_samples(self.output_samples)

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def records(self) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        for (module_name, role), stats in sorted(self.stats.items()):
            item = {
                "label": self.label,
                "module_name": module_name,
                "module_type": self.modules[module_name].__class__.__name__,
                "role": role,
                "statistics": stats.summary(self.config.histogram_bins),
                "quantization": self._metadata_for.get((module_name, role), {}),
                "forward_ms_avg": (
                    sum(self.times.get(module_name, [])) / len(self.times[module_name])
                    if self.times.get(module_name) else None
                ),
            }
            records.append(item)
        return records


def _batches(inputs: Any) -> Iterable[Any]:
    if isinstance(inputs, Mapping) and "batches" in inputs:
        value = inputs["batches"]
        if isinstance(value, (list, tuple)):
            return value
    return [inputs]


def _call_model(model: Any, batch: Any) -> Any:
    if isinstance(batch, Mapping):
        return model(**batch)
    if isinstance(batch, tuple):
        return model(*batch)
    if isinstance(batch, list):
        return model(*batch)
    return model(batch)


def _first_tensor(value: Any) -> Any:
    torch, _ = _torch()
    if isinstance(value, torch.Tensor):
        tensor = _as_tensor(value)
        return tensor.reshape(-1).to("cpu") if tensor is not None else None
    if isinstance(value, Mapping):
        for item in value.values():
            tensor = _first_tensor(item)
            if tensor is not None:
                return tensor
    if isinstance(value, (list, tuple)):
        for item in value:
            tensor = _first_tensor(item)
            if tensor is not None:
                return tensor
    return None


def _combine_output_samples(samples: Sequence[Any], max_elements: int = 100_000) -> Any:
    torch, _ = _torch()
    if not samples:
        return None
    result = torch.cat([item.reshape(-1).to("cpu") for item in samples if item is not None], dim=0)
    return result[:max_elements]


def _compare_values(left: Any, right: Any) -> Optional[Dict[str, Any]]:
    if left is None or right is None:
        return None
    size = min(int(left.numel()), int(right.numel()))
    if size == 0:
        return None
    left = left[:size]
    right = right[:size]
    delta = left - right
    denom = max(float(left.norm().item()), 1e-12)
    return {
        "samples_compared": size,
        "mse": float((delta * delta).mean().item()),
        "mae": float(delta.abs().mean().item()),
        "max_abs_error": float(delta.abs().max().item()),
        "relative_l2": float(delta.norm().item() / denom),
    }


def _merge_pair(records: List[Dict[str, Any]], float_records: List[Dict[str, Any]]) -> None:
    lookup = {(item["module_name"], item["role"]): item for item in float_records}
    for item in records:
        other = lookup.get((item["module_name"], item["role"]))
        if other is None:
            continue
        left = item["statistics"]
        right = other["statistics"]
        # Distribution summary comparison remains useful even when no aligned tensors
        # are retained: compare center and range, and make the limitation explicit.
        item["float_statistics"] = right
        item["comparison"] = {
            "mean_delta": _delta(left.get("mean"), right.get("mean")),
            "std_delta": _delta(left.get("std"), right.get("std")),
            "min_delta": _delta(left.get("min"), right.get("min")),
            "max_delta": _delta(left.get("max"), right.get("max")),
            "abs_max_ratio": _ratio(left.get("abs_max"), right.get("abs_max")),
            "note": "Summary comparison; use golden output or aligned tensor capture for exact error.",
        }
        item["anomaly"] = classify_anomaly(item)


def _delta(left: Optional[float], right: Optional[float]) -> Optional[float]:
    if left is None or right is None:
        return None
    return float(left - right)


def _ratio(left: Optional[float], right: Optional[float]) -> Optional[float]:
    return float(left / right) if left is not None and right not in (None, 0) else None


def _svg_chart(records: Sequence[Dict[str, Any]], title: str, metric: str, width: int = 1000, height: int = 300) -> str:
    chosen = list(records)
    chosen = sorted(chosen, key=lambda item: abs(item["statistics"].get(metric) or 0), reverse=True)[:20]
    if not chosen:
        return f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}"><text x="20" y="30">{html.escape(title)}: no data</text></svg>'
    values = [abs(float(item["statistics"].get(metric) or 0)) for item in chosen]
    maximum = max(values) or 1.0
    margin_left, margin_bottom = 250, 45
    plot_width = width - margin_left - 20
    bar_height = max(8, min(20, (height - margin_bottom) / len(chosen) - 4))
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
             f'<rect width="100%" height="100%" fill="white"/><text x="20" y="25" font-size="18" font-weight="bold">{html.escape(title)}</text>']
    for index, (item, value) in enumerate(zip(chosen, values)):
        y = 42 + index * (bar_height + 5)
        label = f'{item["module_name"]}:{item["role"]}'
        bar_width = plot_width * value / maximum
        parts.append(f'<text x="{margin_left - 8}" y="{y + bar_height - 2}" text-anchor="end" font-size="11">{html.escape(label[:38])}</text>')
        parts.append(f'<rect x="{margin_left}" y="{y}" width="{bar_width:.2f}" height="{bar_height:.2f}" fill="#2F6B9A"/>')
        parts.append(f'<text x="{margin_left + bar_width + 5:.2f}" y="{y + bar_height - 2}" font-size="11">{value:.5g}</text>')
    parts.append("</svg>")
    return "".join(parts)


def _svg_histogram(record: Dict[str, Any], width: int = 900, height: int = 360) -> str:
    stats = record["statistics"]
    histogram = stats.get("histogram", {})
    counts = histogram.get("counts", [])
    edges = histogram.get("edges", [])
    title = f'{record["module_name"]} / {record["role"]}'
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
             '<rect width="100%" height="100%" fill="white"/>',
             f'<text x="20" y="26" font-size="18" font-weight="bold">{html.escape(title)}</text>']
    if not counts or len(edges) < 2:
        parts.append('<text x="20" y="60">No finite samples</text></svg>')
        return "".join(parts)
    plot_x, plot_y, plot_w, plot_h = 55, 45, width - 75, height - 90
    max_count = max(counts) or 1
    bar_w = plot_w / len(counts)
    for index, count in enumerate(counts):
        bar_h = plot_h * count / max_count
        x = plot_x + index * bar_w
        y = plot_y + plot_h - bar_h
        parts.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{max(1, bar_w - 1):.2f}" height="{bar_h:.2f}" fill="#2E8B68"/>')
    parts.append(f'<text x="{plot_x}" y="{height - 20}" font-size="11">{edges[0]:.5g}</text>')
    parts.append(f'<text x="{plot_x + plot_w - 45}" y="{height - 20}" font-size="11">{edges[-1]:.5g}</text>')
    parts.append(f'<text x="{plot_x + plot_w / 2 - 35}" y="{height - 20}" font-size="11">mean={stats.get("mean")!s}, std={stats.get("std")!s}</text>')
    parts.append("</svg>")
    return "".join(parts)


def _write_csv(path: Path, records: Sequence[Dict[str, Any]]) -> None:
    fields = [
        "label", "module_name", "module_type", "role", "count", "finite_count", "min", "max",
        "abs_max", "mean", "std", "p50", "p90", "p99", "p99.9", "zero_ratio", "nan_count",
        "inf_count", "forward_ms_avg", "quantization", "comparison", "anomaly",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for item in records:
            stats = item["statistics"]
            writer.writerow({
                "label": item["label"], "module_name": item["module_name"],
                "module_type": item["module_type"], "role": item["role"],
                **{key: stats.get(key) for key in fields if key in stats},
                "forward_ms_avg": item.get("forward_ms_avg"),
                "quantization": json.dumps(item.get("quantization", {}), ensure_ascii=False),
                "comparison": json.dumps(item.get("comparison", {}), ensure_ascii=False),
                "anomaly": json.dumps(item.get("anomaly", {}), ensure_ascii=False),
            })


def _write_report(path: Path, records: Sequence[Dict[str, Any]], summary: Dict[str, Any]) -> None:
    # Assets are inlined so the generated report needs neither a server nor CDN.
    assets = Path(__file__).with_name("report_assets")
    payload = json.dumps(_jsonable({"records": list(records), "summary": summary}), ensure_ascii=False)
    payload = payload.replace("<", "\\u003c")
    template = (assets / "report.html").read_text(encoding="utf-8")
    template = template.replace("__STYLE__", (assets / "report.css").read_text(encoding="utf-8"))
    script = (assets / "histogram.js").read_text(encoding="utf-8") + "\n" + (assets / "report.js").read_text(encoding="utf-8")
    template = template.replace("__SCRIPT__", script)
    template = template.replace("__GENERATED__", html.escape(str(summary.get("generated_at", ""))))
    path.write_text(template.replace("__PAYLOAD__", payload), encoding="utf-8")


def _load_inputs(path: Optional[str], input_shape: Optional[Sequence[int]], input_dtype: str, seed: int) -> Any:
    torch, _ = _torch()
    if path:
        input_path = Path(path)
        if not input_path.is_file():
            raise AnalysisError(f"Input sample file does not exist: {input_path}")
        suffix = input_path.suffix.lower()
        if suffix in (".npy", ".npz"):
            try:
                import numpy as np
            except ImportError as exc:
                raise AnalysisError("NumPy is required to load .npy/.npz inputs") from exc
            loaded = np.load(str(input_path), allow_pickle=False)
            if suffix == ".npz":
                return {key: torch.from_numpy(value) for key, value in loaded.items()}
            return torch.from_numpy(loaded)
        try:
            return torch.load(str(input_path), map_location="cpu")
        except Exception as exc:
            raise AnalysisError(f"Failed to load input samples '{input_path}': {exc}") from exc
    if not input_shape:
        raise AnalysisError("Provide --inputs or --input-shape when no sample file is available")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    try:
        dtype = getattr(torch, input_dtype)
    except AttributeError as exc:
        raise AnalysisError(f"Unsupported torch dtype: {input_dtype}") from exc
    return torch.randn(tuple(input_shape), generator=generator, dtype=dtype)


def _build_model(factory: Callable[..., Any], kwargs: Mapping[str, Any]) -> Any:
    try:
        model = factory(**kwargs)
    except TypeError:
        if kwargs:
            raise
        model = factory()
    if not hasattr(model, "named_modules") or not hasattr(model, "eval"):
        raise AnalysisError("Model factory must return a torch.nn.Module")
    return model


def analyze(
    model_factory: str,
    checkpoint: Optional[str] = None,
    inputs_path: Optional[str] = None,
    output_dir: str = "quant_analysis_report",
    config: Optional[AnalysisConfig] = None,
    float_checkpoint: Optional[str] = None,
    input_shape: Optional[Sequence[int]] = None,
    input_dtype: str = "float32",
    factory_kwargs: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Run analysis and return the generated summary dictionary."""
    torch, _ = _torch()
    cfg = config or AnalysisConfig(output_dir=output_dir)
    cfg.output_dir = output_dir or cfg.output_dir
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    device = _device(cfg.device)
    factory = _import_callable(model_factory)
    kwargs = dict(factory_kwargs or {})
    inputs = _load_inputs(inputs_path, input_shape, input_dtype, cfg.seed)
    inputs = _move(inputs, device)

    quant_model = _build_model(factory, kwargs).to(device)
    quant_load = _load_checkpoint(quant_model, checkpoint, cfg.strict)
    quant_observer = _ModelObserver(quant_model, cfg, cfg.model_label)
    try:
        quant_output = quant_observer.observe(inputs, cfg.max_batches)
        records = quant_observer.records()
    finally:
        quant_observer.close()

    float_records: List[Dict[str, Any]] = []
    float_load: Dict[str, Any] = {"path": None, "missing_keys": [], "unexpected_keys": []}
    float_output = None
    if float_checkpoint:
        float_model = _build_model(factory, kwargs).to(device)
        float_load = _load_checkpoint(float_model, float_checkpoint, cfg.strict)
        float_observer = _ModelObserver(float_model, cfg, cfg.float_label)
        try:
            float_output = float_observer.observe(inputs, cfg.max_batches)
            float_records = float_observer.records()
        finally:
            float_observer.close()
        _merge_pair(records, float_records)
        # Keep both runs in the exported dataset.  Quantized records carry the
        # pairwise summary in ``comparison`` while float records remain useful
        # as an independently filterable run in the HTML report.
        records.extend(float_records)

    for item in records:
        item.setdefault("anomaly", classify_anomaly(item))

    result_dir = Path(cfg.output_dir)
    figures_dir = result_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    hist_records = records[: cfg.max_histograms]
    summary: Dict[str, Any] = {
        "tool": "standalone-quant-analysis",
        "schema_version": 2,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
        "batches": min(cfg.max_batches, len(list(_batches(inputs)))),
        "module_count": len({item["module_name"] for item in records}),
        "record_count": len(records),
        "model_factory": model_factory,
        "runs": [
            {"label": cfg.model_label, "checkpoint": quant_load},
            *([{"label": cfg.float_label, "checkpoint": float_load}] if float_checkpoint else []),
        ],
        "config": _jsonable(asdict(cfg)),
        "anomaly_rules": ANOMALY_RULES,
        "histogram_records": hist_records,
        "output_comparison": _compare_values(_first_tensor(quant_output), _first_tensor(float_output)) if float_checkpoint else None,
    }
    summary["histogram_records"] = [
        {"label": item["label"], "module_name": item["module_name"], "role": item["role"], "statistics": item["statistics"]}
        for item in hist_records
    ]
    (result_dir / "summary.json").write_text(json.dumps(_jsonable(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    (result_dir / "layer_records.json").write_text(json.dumps(_jsonable(records), ensure_ascii=False, indent=2), encoding="utf-8")
    _write_csv(result_dir / "layer_metrics.csv", records)
    weight_records = [item for item in records if item["role"] == "weight"]
    (figures_dir / "weight_abs_max.svg").write_text(_svg_chart(weight_records, "Weight abs_max by layer", "abs_max"), encoding="utf-8")
    activation_records = [item for item in records if item["role"].startswith("activation")]
    (figures_dir / "activation_abs_max.svg").write_text(_svg_chart(activation_records, "Activation abs_max by layer", "abs_max"), encoding="utf-8")
    for item in hist_records:
        filename = _safe_name(item["label"] + "_" + item["module_name"] + "_" + item["role"]) + ".svg"
        (figures_dir / filename).write_text(_svg_histogram(item), encoding="utf-8")
    _write_report(result_dir / "report.html", records, summary)
    return summary

"""Command line interface for :mod:`tools.quant_analysis`."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

from .core import AnalysisConfig, AnalysisError, analyze


def _config_file(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    config_path = Path(path)
    if not config_path.is_file():
        raise AnalysisError(f"Config file does not exist: {config_path}")
    text = config_path.read_text(encoding="utf-8")
    if config_path.suffix.lower() == ".json":
        value = json.loads(text)
    else:
        try:
            import yaml
        except ImportError as exc:
            raise AnalysisError("PyYAML is required for YAML config files") from exc
        value = yaml.safe_load(text)
    if not isinstance(value, dict):
        raise AnalysisError("Config file must contain a mapping")
    return value


def _value(args: argparse.Namespace, values: Dict[str, Any], name: str, default: Any = None) -> Any:
    cli_value = getattr(args, name)
    return cli_value if cli_value is not None else values.get(name, default)


def _shape(value: Any) -> Optional[list[int]]:
    if value is None:
        return None
    if isinstance(value, str):
        try:
            shape = [int(item.strip()) for item in value.split(",") if item.strip()]
            if not shape or any(item <= 0 for item in shape):
                raise ValueError
            return shape
        except ValueError as exc:
            raise AnalysisError("input_shape must look like 1,3,224,224") from exc
    shape = [int(item) for item in value]
    if not shape or any(item <= 0 for item in shape):
        raise AnalysisError("input_shape dimensions must be positive")
    return shape


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="quant-analysis",
        description="Analyze PyTorch weight and activation distributions without importing linger.",
        epilog="Example: python -m tools.quant_analysis --model-factory model.py:build_model --checkpoint qat.pt --inputs calibration.pt --output-dir artifacts/quantviz",
    )
    parser.add_argument("--config", help="Optional YAML/JSON config; command-line values override it")
    parser.add_argument("--model-factory", dest="model_factory")
    parser.add_argument("--checkpoint")
    parser.add_argument("--float-checkpoint")
    parser.add_argument("--inputs", dest="inputs_path")
    parser.add_argument("--input-shape", help="Fallback shape, e.g. 1,3,224,224")
    parser.add_argument("--input-dtype", default=None)
    parser.add_argument("--factory-kwargs", help="JSON object passed to the model factory")
    parser.add_argument("--output-dir")
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--histogram-bins", type=int)
    parser.add_argument("--max-histograms", type=int)
    parser.add_argument("--device")
    parser.add_argument("--include", action="append", default=None, help="Module glob to include; repeatable")
    parser.add_argument("--exclude", action="append", default=None, help="Module glob to exclude; repeatable")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--non-strict", action="store_true", help="Allow missing/unexpected checkpoint keys")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        values = _config_file(args.config)
        factory = _value(args, values, "model_factory")
        if not factory:
            raise AnalysisError("--model-factory is required")
        output_dir = _value(args, values, "output_dir", "quant_analysis_report")
        cfg = AnalysisConfig(
            output_dir=output_dir,
            max_batches=int(_value(args, values, "max_batches", 32)),
            max_samples=int(_value(args, values, "max_samples", 100_000)),
            histogram_bins=int(_value(args, values, "histogram_bins", 64)),
            max_histograms=int(_value(args, values, "max_histograms", 12)),
            device=_value(args, values, "device", "auto"),
            include=_value(args, values, "include", []) or [],
            exclude=_value(args, values, "exclude", []) or [],
            seed=int(_value(args, values, "seed", 1234)),
            strict=not args.non_strict,
        )
        for field_name in ("max_batches", "max_samples", "histogram_bins", "max_histograms"):
            if getattr(cfg, field_name) <= 0:
                raise AnalysisError(f"{field_name} must be positive")
        kwargs_value = args.factory_kwargs if args.factory_kwargs is not None else values.get("factory_kwargs", {})
        if isinstance(kwargs_value, str):
            kwargs_value = json.loads(kwargs_value)
        if not isinstance(kwargs_value, dict):
            raise AnalysisError("factory_kwargs must be a JSON object")
        summary = analyze(
            model_factory=factory,
            checkpoint=_value(args, values, "checkpoint"),
            float_checkpoint=_value(args, values, "float_checkpoint"),
            inputs_path=_value(args, values, "inputs_path"),
            input_shape=_shape(_value(args, values, "input_shape")),
            input_dtype=_value(args, values, "input_dtype", "float32"),
            output_dir=output_dir,
            config=cfg,
            factory_kwargs=kwargs_value,
        )
        print(f"Analysis complete: {output_dir}")
        print(f"  report:  {Path(output_dir) / 'report.html'}")
        print(f"  records: {summary['record_count']} ({summary['module_count']} modules)")
        return 0
    except AnalysisError as exc:
        parser.error(str(exc))
    except Exception as exc:
        parser.error(f"unexpected error: {exc}")
    return 2

"""Beginner-friendly entry point for the standalone quantization analyzer."""

try:
    from tools.quant_analysis.cli import main
except ModuleNotFoundError:  # Direct execution: ``python tools/analyze_quantization.py``.
    from quant_analysis.cli import main


if __name__ == "__main__":
    raise SystemExit(main())

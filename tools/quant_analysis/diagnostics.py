"""Explainable distribution screening; no quantization-framework assumptions."""

from typing import Any, Dict, List

# Stable thresholds are exported with every run so warnings remain reproducible.
ANOMALY_RULES = {
    "version": 1,
    "ratio_low": 0.5,
    "ratio_high": 2.0,
    "mean_shift_std": 1.0,
    "zero_ratio_increase": 0.25,
    "description": (
        "NaN/Inf 或非空张量无有限值：数值错误。相对浮点参考的 abs_max 或标准差比值 "
        "< 0.5 或 > 2、均值偏移 > 1 个浮点标准差、零值比例增加 ≥ 25 个百分点：待复核。"
        "浮点参考为全零或常量时，量化结果出现新的非零范围或方差也需复核。阈值不改变数值一致性容差。"
    ),
}


def classify_anomaly(item: Dict[str, Any]) -> Dict[str, Any]:
    """Return evidence, not an assertion that quantization caused the issue.

    Numerical errors can be detected without a reference. Distribution drift
    requires a matched reference; natural sparsity/constant tensors alone do
    not imply failure. No saturation claim is made from a histogram's endpoints.
    """
    stats = item.get("statistics", {})
    reference = item.get("float_statistics") or {}
    reasons: List[str] = []
    critical = False
    if (stats.get("nan_count") or 0) + (stats.get("inf_count") or 0):
        reasons.append(f"NaN={stats.get('nan_count', 0)}，Inf={stats.get('inf_count', 0)}（阈值：任意非有限值）")
        critical = True
    if stats.get("count", 0) > 0 and stats.get("finite_count", 0) == 0:
        reasons.append("非空张量没有有限值")
        critical = True
    if stats.get("finite_count", 0) and reference.get("finite_count", 0):
        for field in ("abs_max", "std"):
            q, f = stats.get(field), reference.get(field)
            if q is None or f is None:
                continue
            if f > 0:
                ratio = q / f
                if ratio < ANOMALY_RULES["ratio_low"] or ratio > ANOMALY_RULES["ratio_high"]:
                    reasons.append(f"{field} 比值={ratio:.4g}（量化={q:.4g}，浮点={f:.4g}；阈值 < 0.5 或 > 2）")
            elif q > 0:
                reasons.append(f"浮点 {field}=0，量化 {field}={q:.4g}")
        qs, fs = stats.get("mean"), reference.get("mean")
        std = reference.get("std")
        if qs is not None and fs is not None and std is not None and std > 0:
            shift = abs(qs - fs) / std
            if shift > ANOMALY_RULES["mean_shift_std"]:
                reasons.append(f"均值偏移={shift:.4g} 个浮点标准差（阈值 > 1）")
        qz, fz = stats.get("zero_ratio"), reference.get("zero_ratio")
        if qz is not None and fz is not None and qz >= fz + ANOMALY_RULES["zero_ratio_increase"]:
            reasons.append(f"零值比例增加 {(qz - fz) * 100:.2f} 个百分点（量化={qz:.2%}，浮点={fz:.2%}；阈值 ≥ 25）")
    return {
        "flag": bool(reasons),
        "severity": "critical" if critical else "warning" if reasons else "ok",
        "reasons": reasons,
        "reference_available": bool(reference),
    }

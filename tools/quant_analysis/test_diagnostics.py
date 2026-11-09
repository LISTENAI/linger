"""Run with: python -m unittest tools.quant_analysis.test_diagnostics"""

import copy
import json
import tempfile
import unittest
from pathlib import Path

from .core import _merge_pair, _write_csv, _write_report
from .diagnostics import classify_anomaly


def record(label="quantized", **overrides):
    stats = dict(count=100, finite_count=100, nan_count=0, inf_count=0,
                 min=-1.0, max=1.0, abs_max=1.0, mean=0.0, std=0.5,
                 zero_ratio=0.1, histogram={"edges": [-1.0, 0.0, 1.0], "counts": [50, 50]})
    stats.update(overrides)
    return dict(label=label, module_name="linear", module_type="Linear", role="weight", statistics=stats)


class DiagnosticsTest(unittest.TestCase):
    def pair(self, q, f=None):
        _merge_pair([q], [f if f is not None else record("float")])
        return q["anomaly"]

    def test_reference_and_natural_sparsity_are_not_flagged(self):
        for q, f in [(record(), record("float")),
                     (record(std=0, abs_max=0, zero_ratio=1), record("float", std=0, abs_max=0, zero_ratio=1)),
                     (record(zero_ratio=0.9999), record("float", zero_ratio=0.9999))]:
            self.assertFalse(self.pair(q, f)["flag"])
        self.assertFalse(classify_anomaly(record(zero_ratio=1, std=0))["flag"])

    def test_numerical_errors_with_and_without_reference(self):
        for r in [record(nan_count=1), record(inf_count=1), record(finite_count=0)]:
            self.assertEqual(classify_anomaly(r)["severity"], "critical")
            self.assertEqual(self.pair(r)["severity"], "critical")

    def test_ratio_thresholds_are_strict(self):
        for value in (0.5, 2.0):
            self.assertFalse(self.pair(record(abs_max=value))["flag"])
        for value in (0.49, 2.01):
            a = self.pair(record(abs_max=value))
            self.assertTrue(a["flag"])
            self.assertTrue(any('abs_max' in s for s in a["reasons"]))

    def test_collapse_shift_and_new_variance(self):
        for r in [record(std=0), record(mean=0.6), record(zero_ratio=0.35)]:
            self.assertEqual(self.pair(r)["severity"], "warning")
        self.assertTrue(self.pair(record(), record("float", abs_max=0, std=0))["flag"])

    def test_no_pair_and_empty_input(self):
        q = record()
        _merge_pair([q], [])
        self.assertNotIn("float_statistics", q)
        self.assertFalse(classify_anomaly(record(count=0, finite_count=0))["flag"])

    def test_report_is_offline_and_data_is_script_safe(self):
        r = record()
        r["module_name"] = '</script><img src=x onerror="bad()">'
        r["anomaly"] = classify_anomaly(r)
        original = copy.deepcopy(r)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.html"
            _write_report(path, [r], {"generated_at": "test", "config": {}})
            content = path.read_text()
            self.assertNotIn('id="metadata"', content)
            self.assertNotIn('src="http', content)
            self.assertNotIn('id="cdf"', content)
            for category in ('weight', 'bias', 'activation'):
                self.assertIn(f'id="{category}-gallery"', content)
                self.assertIn(f'id="{category}-window"', content)
                self.assertIn(f'data-category="{category}"', content)
            self.assertIn('id="layer-search"', content)
            self.assertNotIn('<select', content)
            data = content.split('<script id="analysis-data" type="application/json">')[1].split('</script>')[0]
            self.assertNotIn('<img', data)
            self.assertEqual(json.loads(data)["records"][0], original)
            csv = Path(directory) / "records.csv"
            _write_csv(csv, [r])
            self.assertIn('anomaly', csv.read_text().splitlines()[0])
        self.assertEqual(original, r)


if __name__ == "__main__":
    unittest.main()

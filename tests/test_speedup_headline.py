"""dashboard.results_io E5 speedup headline helpers (training-N speedup, range over N, wide table)."""

import pandas as pd
import pytest

rio = pytest.importorskip("dashboard.results_io")


def _sp():
    rows = []
    for ps in ("fwd", "fwd_bwd"):
        for n, s in ((1024, 3.0), (8192, 1.5), (65536, 0.9)):
            rows.append({"sweep": "tokens", "variant": "switch", "pass": ps, "n_tokens": n, "n_experts": 16,
                         "loop_ms": s, "batched_ms": 1.0, "speedup": s})
    rows.append({"sweep": "tokens", "variant": "switch", "pass": "fwd", "n_tokens": 8192, "n_experts": 64,
                 "loop_ms": 9.0, "batched_ms": 1.0, "speedup": 9.0})  # other E: must be excluded
    return pd.DataFrame(rows)


def test_headline_train_n_and_range():
    h = rio.speedup_headline(_sp(), 8192, {"switch": 16})
    r = h[(h["variant"] == "switch") & (h["pass"] == "fwd")].iloc[0]
    assert r["speedup_train"] == 1.5 and (r["min_speedup"], r["min_n"]) == (0.9, 65536) and (r["max_speedup"], r["max_n"]) == (3.0, 1024)
    assert "1.50x at N=8,192 (training size); range 0.90x (N=65,536) to 3.00x (N=1,024)" in rio.speedup_headline_text(h)[0]


def test_headline_train_n_missing_and_empty():
    h = rio.speedup_headline(_sp(), 4096, {"switch": 16})
    assert "not in the sweep" in rio.speedup_headline_text(h)[0]
    assert rio.speedup_headline(pd.DataFrame(), 8192).empty


def test_wide_table():
    w = rio.speedup_vs_n_table(_sp(), {"switch": 16})
    assert list(w["pass"]) == ["fwd", "fwd_bwd"] and w[8192].tolist() == [1.5, 1.5]

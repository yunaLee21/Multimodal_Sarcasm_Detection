import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from analyze_audit_round2 import holm, kappa, load, perm_test, wmean  # noqa: E402


def test_holm_matches_known_values():
    assert np.allclose(holm([0.01, 0.04, 0.03, 0.5]), [0.04, 0.09, 0.09, 0.5])


def test_perm_test_detects_shift_and_respects_strata():
    rng = np.random.default_rng(0)
    strata = np.repeat(["a", "b"], 30)
    needed = np.tile([True] * 10 + [False] * 20, 2)
    y = rng.normal(0, 1, 60) + 2 * needed
    diff, p = perm_test(y, needed, strata, 2000, rng)
    assert diff > 1 and p < 0.01
    # confounded by stratum only: needed all in stratum b, which has a higher mean -> no within-stratum signal possible
    needed2 = strata == "b"
    _, p2 = perm_test(y * 0 + (strata == "b"), needed2, strata, 500, rng)
    assert p2 == 1.0


def test_perm_test_degenerate():
    d, p = perm_test(np.ones(5), np.ones(5, bool), np.zeros(5), 10, np.random.default_rng(0))
    assert np.isnan(d) and np.isnan(p)


def test_kappa_and_wmean():
    assert kappa(["co", "khong", "co"], ["co", "khong", "co"]) == 1.0
    assert np.isnan(kappa(["co", "co"], ["co", "co"]))
    assert wmean([1, 0], np.array([3.0, 1.0])) == 0.75


def test_load_repairs_commas_and_shift(tmp_path):
    f = tmp_path / "s.csv"
    f.write_text("audit_id,caption,sarcasm,confidence,evidence\n"
                 "R001,cap,co,3,a,b,c\n"
                 "R002,cap,,,,khong,2,x\n", encoding="utf-8-sig")
    s = load(f)
    assert s["R001"]["evidence"] == "a,b,c"
    assert s["R002"]["sarcasm"] == "khong" and s["R002"]["evidence"] == "x"

"""
Regression tests for matching quality, using the labeled eval set.
These guard against silent quality degradation — e.g. an embedding
model change, a threshold tweak, or a guard refactor that quietly
breaks ranking or mismatch rejection without anyone noticing until
the demo.

Requires data/eval/labels.json to already exist (run
data/eval/build_labels.py first if not).
"""
from data.eval.run_eval import compute_top1_precision, compute_mismatch_rejection_rate


def test_top1_precision_meets_minimum_bar():
    precision = compute_top1_precision()
    assert precision >= 0.90, (
        f"Top-1 precision dropped to {precision:.2%}, below the 90% floor. "
        f"This likely means a ranking/embedding change regressed match quality."
    )


def test_mismatch_rejection_meets_minimum_bar():
    rejection_rate = compute_mismatch_rejection_rate()
    assert rejection_rate >= 0.90, (
        f"Mismatch rejection dropped to {rejection_rate:.2%}, below the 90% floor. "
        f"This likely means a guard/threshold change regressed safety."
    )
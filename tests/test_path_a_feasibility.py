"""
tests/test_path_a_feasibility.py
Unit tests verifying the mathematical integrity of the Path A DSL search space,
total-order comparator injectivity, dual task ranks, and independent task references.
"""
from typing import List, Any
import pytest
from scripts.path_a_feasibility_check import (
    enumerate_space,
    audit_task_ranks,
    TASKS,
    ASTNode,
)


def test_search_space_depth_counts():
    """Verify exact count of well-typed programs at each depth up to D <= 3."""
    all_programs = enumerate_space(max_depth=3)
    
    d0 = sum(1 for p in all_programs if p.depth() == 0)
    d1 = sum(1 for p in all_programs if p.depth() == 1)
    d2 = sum(1 for p in all_programs if p.depth() == 2)
    d3 = sum(1 for p in all_programs if p.depth() == 3)

    assert d0 == 1, f"Expected 1 Depth 0 program, got {d0}"
    assert d1 == 18, f"Expected 18 Depth 1 programs, got {d1}"
    assert d2 == 252, f"Expected 252 Depth 2 programs, got {d2}"
    assert d3 == 3528, f"Expected 3528 Depth 3 programs, got {d3}"
    assert len(all_programs) == 3799, f"Expected 3799 total programs, got {len(all_programs)}"


def test_comparator_injectivity():
    """Verify that canonical_key() is strictly injective across all 3,799 programs (zero ties)."""
    all_programs = enumerate_space(max_depth=3)
    keys = [p.canonical_key() for p in all_programs]
    unique_keys = set(keys)

    assert len(keys) == 3799
    assert len(unique_keys) == 3799, f"Comparator collision detected! {len(unique_keys)} unique out of {len(keys)}"
    
    # Verify strict ascending sort order
    for i in range(len(keys) - 1):
        assert keys[i] < keys[i + 1], f"Monotonicity violation at index {i}: {keys[i]} >= {keys[i+1]}"


def test_saved_task_ranks():
    """Verify that audited syntactic and dedup ranks match logs/path_a_feasibility_20261026.txt."""
    all_programs = enumerate_space(max_depth=3)
    ranks = audit_task_ranks(all_programs)

    expected_ranks = {
        # Calibration Tasks
        "T01_sum": (17, 16),
        "T02_count": (2, 2),
        "T03_double_all": (8, 7),
        "T04_add_one": (4, 3),
        "T05_filter_positive": (11, 10),
        "T06_filter_evens": (14, 13),
        "T07_count_positive": (28, 20),
        "T08_sum_doubled": (235, 117),
        "T09_add_two_all": (5, 4),
        "T10_filter_ge0": (12, 11),
        # Held-Out Tasks
        "H01_filter_odds": (15, 14),
        "H02_double_evens": (115, 74),
        "H03_add_two_positives": (70, 39),
        "H04_count_evens": (31, 23),
        "H05_count_odds": (32, 24),
        "H06_sum_evens": (241, 119),
        "H07_sum_positives": (238, 116),
        "H08_double_positive_evens": (1571, 395),
        "H09_sum_doubled_evens": (3293, 543),
        "H10_sum_add1_evens": (3237, 503),
    }

    for task_name, (exp_syn, exp_dedup) in expected_ranks.items():
        actual_syn = ranks[task_name]["syntactic_rank"]
        actual_dedup = ranks[task_name]["dedup_rank"]
        assert actual_syn == exp_syn, f"{task_name} syntactic rank mismatch: expected {exp_syn}, got {actual_syn}"
        assert actual_dedup == exp_dedup, f"{task_name} dedup rank mismatch: expected {exp_dedup}, got {actual_dedup}"


def test_independent_task_reference_validation():
    """Verify that expected outputs in TASKS match an independent pure-Python reference implementation."""
    reference_functions = {
        "T01_sum": lambda xs: sum(xs),
        "T02_count": lambda xs: len(xs),
        "T03_double_all": lambda xs: [x * 2 for x in xs],
        "T04_add_one": lambda xs: [x + 1 for x in xs],
        "T05_filter_positive": lambda xs: [x for x in xs if x > 0],
        "T06_filter_evens": lambda xs: [x for x in xs if x % 2 == 0],
        "T07_count_positive": lambda xs: len([x for x in xs if x > 0]),
        "T08_sum_doubled": lambda xs: sum([x * 2 for x in xs]),
        "T09_add_two_all": lambda xs: [x + 2 for x in xs],
        "T10_filter_ge0": lambda xs: [x for x in xs if x >= 0],
        "H01_filter_odds": lambda xs: [x for x in xs if x % 2 != 0],
        "H02_double_evens": lambda xs: [x * 2 for x in xs if x % 2 == 0],
        "H03_add_two_positives": lambda xs: [x + 2 for x in xs if x > 0],
        "H04_count_evens": lambda xs: len([x for x in xs if x % 2 == 0]),
        "H05_count_odds": lambda xs: len([x for x in xs if x % 2 != 0]),
        "H06_sum_evens": lambda xs: sum([x for x in xs if x % 2 == 0]),
        "H07_sum_positives": lambda xs: sum([x for x in xs if x > 0]),
        "H08_double_positive_evens": lambda xs: [x * 2 for x in xs if x % 2 == 0 and x > 0],
        "H09_sum_doubled_evens": lambda xs: sum([x * 2 for x in xs if x % 2 == 0]),
        "H10_sum_add1_evens": lambda xs: sum([x + 1 for x in xs if x % 2 == 0]),
    }

    for task_name, test_cases in TASKS.items():
        ref_fn = reference_functions[task_name]
        for inp, expected in test_cases:
            actual = ref_fn(inp)
            assert actual == expected, (
                f"Task reference discrepancy in {task_name} for input {inp}: "
                f"reference computed {actual}, but task expected {expected}"
            )


def test_flat_cost_prior_equivalence():
    """Verify that FlatCostPrior (ln 1/21) behaves identically to UniformPrior on held-out tasks."""
    import math
    from scripts.prior_controls import FlatCostPrior, UniformPrior, search_task, HELDOUT_10_SEQUENCE

    fc = FlatCostPrior()
    unif = UniformPrior()

    assert fc.log_prob(("len", 0, "ListExpr"), "map") == math.log(1.0 / 21.0)
    assert fc.log_prob(("map", 0, "ScalarOp"), "add1") == math.log(1.0 / 21.0)

    # Verify identical execution counts on subset of tasks (H01, H04, H06)
    for t_name in ["H01_filter_odds", "H04_count_evens", "H06_sum_evens"]:
        res_fc = search_task(t_name, TASKS[t_name], fc, budget_k=500)
        res_unif = search_task(t_name, TASKS[t_name], unif, budget_k=500)
        assert res_fc["executions"] == res_unif["executions"], (
            f"Equivalence mismatch on {t_name}: FlatCost={res_fc['executions']}, Uniform={res_unif['executions']}"
        )

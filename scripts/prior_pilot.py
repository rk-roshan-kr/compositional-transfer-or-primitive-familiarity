"""
scripts/prior_pilot.py
Arm 2 Pilot: Count-Based Prior with Best-First Search (beta = 0).
Tests whether transition statistics from the 10 calibration programs reach
held-out tasks H08-H10 within budget K = 500 executions.
"""
import sys
import os
import math
import heapq
import random
from typing import List, Tuple, Dict, Any, Optional
from collections import defaultdict, Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.path_a_feasibility_check import (
    enumerate_space,
    audit_task_ranks,
    TASKS,
    ASTNode,
    InputList,
    LenNode,
    MapNode,
    FilterNode,
    FoldNode,
    SCALAR_OPS,
    PREDICATES,
    FOLD_REDUCERS,
)

# =====================================================================
# 1. Action Vocabulary & Slot Types
# =====================================================================

VALID_ACTIONS = {
    "IntExpr": ["len", "fold"],
    "ListExpr": ["map", "filter"],
    "ScalarOp": ["id", "add1", "add2", "add3", "sub1", "mul2", "mul3"],
    "Predicate": ["eq0", "gt0", "ge0", "lt0", "even", "odd", "mod3_0"],
    "Reducer": ["add", "sub", "mul"],
}

ALL_21_PRIMITIVES = [
    "map", "filter", "fold", "len",
    "id", "add1", "add2", "add3", "sub1", "mul2", "mul3",
    "eq0", "gt0", "ge0", "lt0", "even", "odd", "mod3_0",
    "add", "sub", "mul"
]
assert len(ALL_21_PRIMITIVES) == 21


# =====================================================================
# 2. Canonical AST Reconstruction & Transition Extraction
# =====================================================================

def reconstruct_ast(prims: List[str]) -> ASTNode:
    """Reconstructs ASTNode from a primitive token sequence."""
    if not prims:
        return InputList()
    c = prims[0]
    if c == "len":
        return LenNode(reconstruct_ast(prims[1:]))
    elif c == "map":
        op_name = prims[1]
        fn, rpr, idx = [(fn, rpr, idx) for name, fn, rpr, idx in SCALAR_OPS if name == op_name][0]
        return MapNode(op_name, fn, rpr, idx, reconstruct_ast(prims[2:]))
    elif c == "filter":
        pred_name = prims[1]
        fn, rpr, idx = [(fn, rpr, idx) for name, fn, rpr, idx in PREDICATES if name == pred_name][0]
        return FilterNode(pred_name, fn, rpr, idx, reconstruct_ast(prims[2:]))
    elif c == "fold":
        red_name = prims[1]
        fn, rpr, init, idx = [(fn, rpr, init, idx) for name, fn, rpr, init, idx in FOLD_REDUCERS if name == red_name][0]
        return FoldNode(red_name, fn, rpr, init, idx, reconstruct_ast(prims[2:]))
    raise ValueError(f"Unknown combinator token: {c}")


def get_canonical_transitions(node: ASTNode, root_type: str) -> List[Tuple[Tuple, str]]:
    """Extracts sequence of (ctx, action) transitions in pre-order traversal."""
    transitions = []
    if isinstance(node, LenNode):
        transitions.append(((None, 0, root_type), "len"))
        curr = node.child
        parent_comb = "len"
        arg_pos = 0
        while not isinstance(curr, InputList):
            if isinstance(curr, MapNode):
                transitions.append(((parent_comb, arg_pos, "ListExpr"), "map"))
                transitions.append((("map", 0, "ScalarOp"), curr.op_name))
                parent_comb, arg_pos, curr = "map", 1, curr.child
            elif isinstance(curr, FilterNode):
                transitions.append(((parent_comb, arg_pos, "ListExpr"), "filter"))
                transitions.append((("filter", 0, "Predicate"), curr.pred_name))
                parent_comb, arg_pos, curr = "filter", 1, curr.child
    elif isinstance(node, FoldNode):
        transitions.append(((None, 0, root_type), "fold"))
        transitions.append((("fold", 0, "Reducer"), node.reducer_name))
        curr = node.child
        parent_comb = "fold"
        arg_pos = 1
        while not isinstance(curr, InputList):
            if isinstance(curr, MapNode):
                transitions.append(((parent_comb, arg_pos, "ListExpr"), "map"))
                transitions.append((("map", 0, "ScalarOp"), curr.op_name))
                parent_comb, arg_pos, curr = "map", 1, curr.child
            elif isinstance(curr, FilterNode):
                transitions.append(((parent_comb, arg_pos, "ListExpr"), "filter"))
                transitions.append((("filter", 0, "Predicate"), curr.pred_name))
                parent_comb, arg_pos, curr = "filter", 1, curr.child
    elif isinstance(node, MapNode):
        transitions.append(((None, 0, root_type), "map"))
        transitions.append((("map", 0, "ScalarOp"), node.op_name))
        curr = node.child
        parent_comb = "map"
        arg_pos = 1
        while not isinstance(curr, InputList):
            if isinstance(curr, MapNode):
                transitions.append(((parent_comb, arg_pos, "ListExpr"), "map"))
                transitions.append((("map", 0, "ScalarOp"), curr.op_name))
                parent_comb, arg_pos, curr = "map", 1, curr.child
            elif isinstance(curr, FilterNode):
                transitions.append(((parent_comb, arg_pos, "ListExpr"), "filter"))
                transitions.append((("filter", 0, "Predicate"), curr.pred_name))
                parent_comb, arg_pos, curr = "filter", 1, curr.child
    elif isinstance(node, FilterNode):
        transitions.append(((None, 0, root_type), "filter"))
        transitions.append((("filter", 0, "Predicate"), node.pred_name))
        curr = node.child
        parent_comb = "filter"
        arg_pos = 1
        while not isinstance(curr, InputList):
            if isinstance(curr, MapNode):
                transitions.append(((parent_comb, arg_pos, "ListExpr"), "map"))
                transitions.append((("map", 0, "ScalarOp"), curr.op_name))
                parent_comb, arg_pos, curr = "map", 1, curr.child
            elif isinstance(curr, FilterNode):
                transitions.append(((parent_comb, arg_pos, "ListExpr"), "filter"))
                transitions.append((("filter", 0, "Predicate"), curr.pred_name))
                parent_comb, arg_pos, curr = "filter", 1, curr.child
    return transitions


# =====================================================================
# 3. Smoothed Bigram Prior Model
# =====================================================================

class CountBasedPrior:
    def __init__(self, training_transitions: List[Tuple[Tuple, str]], alpha: float = 1.0):
        self.alpha = alpha
        self.ctx_counts = defaultdict(Counter)
        self.slot_counts = defaultdict(Counter)

        for ctx, act in training_transitions:
            self.ctx_counts[ctx][act] += 1
            slot_type = ctx[2]
            self.slot_counts[slot_type][act] += 1

    def log_prob(self, ctx: Tuple, a: str) -> Tuple[float, str]:
        """
        Returns (ln P(a | ctx), backoff_level_name).
        Level 1: Full context (parent, arg_pos, slot_type)
        Level 2: Backoff to slot_type
        Level 3: Backoff to uniform
        """
        slot_type = ctx[2]
        val_actions = VALID_ACTIONS[slot_type]
        c_ctx = sum(self.ctx_counts[ctx][act] for act in val_actions)

        # Level 1: Full Context
        if c_ctx > 0:
            count = self.ctx_counts[ctx][a]
            p = (count + self.alpha) / (c_ctx + self.alpha * len(val_actions))
            return math.log(p), "level1_context"

        # Level 2: Slot Type Alone
        c_slot = sum(self.slot_counts[slot_type][act] for act in val_actions)
        if c_slot > 0:
            count = self.slot_counts[slot_type][a]
            p = (count + self.alpha) / (c_slot + self.alpha * len(val_actions))
            return math.log(p), "level2_slot"

        # Level 3: Uniform
        p = 1.0 / len(val_actions)
        return math.log(p), "level3_uniform"

    def predict_top1(self, ctx: Tuple) -> Tuple[str, float]:
        """Returns argmax_a P(a | ctx) and its probability."""
        slot_type = ctx[2]
        val_actions = VALID_ACTIONS[slot_type]
        best_a = None
        best_lp = -float("inf")
        for a in val_actions:
            lp, _ = self.log_prob(ctx, a)
            if lp > best_lp:
                best_lp = lp
                best_a = a
        return best_a, math.exp(best_lp)


# =====================================================================
# 4. Gate 1 Diagnostic (50 Synthetic ASTs)
# =====================================================================

def evaluate_gate1_diagnostic(prior: CountBasedPrior) -> Dict[str, Any]:
    all_programs = enumerate_space(max_depth=3)
    ranks = audit_task_ranks(all_programs)

    calib_keys = {all_programs[ranks[t]["syntactic_rank"] - 1].canonical_key() for t in TASKS if t.startswith("T")}
    heldout_keys = {all_programs[ranks[t]["syntactic_rank"] - 1].canonical_key() for t in TASKS if t.startswith("H")}

    forbidden_subtrees = set()
    for t in ["H02_double_evens", "H03_add_two_positives", "H04_count_evens", "H05_count_odds", "H06_sum_evens", "H07_sum_positives"]:
        p = all_programs[ranks[t]["syntactic_rank"] - 1]
        if isinstance(p, (MapNode, FilterNode)):
            forbidden_subtrees.add(p.canonical_key())
        elif hasattr(p, "child") and isinstance(p.child, (MapNode, FilterNode)):
            forbidden_subtrees.add(p.child.canonical_key())

    for t in ["H08_double_positive_evens", "H09_sum_doubled_evens", "H10_sum_add1_evens"]:
        p = all_programs[ranks[t]["syntactic_rank"] - 1]
        if hasattr(p, "child") and p.child.depth() == 2:
            forbidden_subtrees.add(p.child.canonical_key())

    pool_d1 = sorted([p for p in all_programs if p.depth() == 1 and p.canonical_key() not in calib_keys and p.canonical_key() not in heldout_keys], key=lambda x: x.canonical_key())
    pool_d2 = sorted([p for p in all_programs if p.depth() == 2 and p.canonical_key() not in calib_keys and p.canonical_key() not in heldout_keys], key=lambda x: x.canonical_key())
    pool_d3 = sorted([p for p in all_programs if p.depth() == 3 and p.canonical_key() not in heldout_keys and p.child.canonical_key() not in forbidden_subtrees], key=lambda x: x.canonical_key())

    rng = random.Random(20261026)
    s1 = rng.sample(pool_d1, 5)
    s2 = rng.sample(pool_d2, 20)
    s3 = rng.sample(pool_d3, 25)
    sample_50 = s1 + s2 + s3
    assert len(sample_50) == 50

    depth_stats = {1: {"correct": 0, "total": 0, "chance": 0.0},
                   2: {"correct": 0, "total": 0, "chance": 0.0},
                   3: {"correct": 0, "total": 0, "chance": 0.0}}

    overall_correct = 0
    overall_total = 0
    overall_chance = 0.0

    for p in sample_50:
        d = p.depth()
        root_type = "IntExpr" if isinstance(p, (LenNode, FoldNode)) else "ListExpr"
        transitions = get_canonical_transitions(p, root_type)

        for ctx, target_act in transitions:
            slot_type = ctx[2]
            val_actions = VALID_ACTIONS[slot_type]
            chance_val = 1.0 / len(val_actions)

            pred_act, _ = prior.predict_top1(ctx)
            is_correct = 1 if (pred_act == target_act) else 0

            depth_stats[d]["correct"] += is_correct
            depth_stats[d]["total"] += 1
            depth_stats[d]["chance"] += chance_val

            overall_correct += is_correct
            overall_total += 1
            overall_chance += chance_val

    assert overall_total == 239

    return {
        "depth_stats": depth_stats,
        "overall_correct": overall_correct,
        "overall_total": overall_total,
        "overall_acc": overall_correct / overall_total,
        "overall_chance": overall_chance / overall_total,
    }


# =====================================================================
# 5. Best-First Search for Arm 2 (Learned Prior, beta = 0)
# =====================================================================

def search_task(task_name: str, prior: CountBasedPrior, budget_k: int = 500,
                max_expansions: int = 100000, lambda_val: float = 0.05) -> Dict[str, Any]:
    test_cases = TASKS[task_name]
    is_scalar_out = isinstance(test_cases[0][1], int)
    root_type = "IntExpr" if is_scalar_out else "ListExpr"

    heap = []
    initial_slot = (None, 0, root_type)

    # If output is List, include InputList() (xs) as a depth 0 complete candidate
    if not is_scalar_out:
        ast_zero = InputList()
        heapq.heappush(heap, (-0.0, ast_zero.canonical_key(), [], [], 0))

    # Push root partial program
    heapq.heappush(heap, (-0.0, (0, 0, ()), [], [initial_slot], 0))

    executions = 0
    expansions = 0
    seen_canonical = set()
    solved = False
    solution_render = ""
    backoffs = Counter()
    hit_expansion_cap = False

    while heap and executions < budget_k and expansions < max_expansions:
        neg_score, key, prims, slots, depth = heapq.heappop(heap)
        score = -neg_score

        if not slots:
            # Complete executable program
            ast_node = reconstruct_ast(prims)
            c_key = ast_node.canonical_key()
            if c_key in seen_canonical:
                continue
            seen_canonical.add(c_key)

            executions += 1

            # Evaluate on visible cases
            passed = True
            for inp, exp in test_cases:
                try:
                    if ast_node.eval(inp) != exp:
                        passed = False
                        break
                except Exception:
                    passed = False
                    break

            if passed:
                solved = True
                solution_render = ast_node.render()
                break
        else:
            # Partial program expansion
            expansions += 1
            slot = slots[0]
            rem_slots = slots[1:]
            slot_type = slot[2]

            if slot_type == "IntExpr":
                for act in ["len", "fold"]:
                    lp, lvl = prior.log_prob(slot, act)
                    backoffs[lvl] += 1
                    new_score = score + lp - lambda_val
                    new_slots = [("len", 0, "ListExpr")] + rem_slots if act == "len" else [("fold", 0, "Reducer"), ("fold", 1, "ListExpr")] + rem_slots
                    new_prims = prims + [act]
                    t_key = (depth + 1, len(new_prims), tuple(new_prims))
                    heapq.heappush(heap, (-new_score, t_key, new_prims, new_slots, depth + 1))

            elif slot_type == "Reducer":
                for act in ["add", "sub", "mul"]:
                    lp, lvl = prior.log_prob(slot, act)
                    backoffs[lvl] += 1
                    new_score = score + lp - lambda_val
                    new_prims = prims + [act]
                    t_key = (depth, len(new_prims), tuple(new_prims))
                    heapq.heappush(heap, (-new_score, t_key, new_prims, rem_slots, depth))

            elif slot_type == "ScalarOp":
                for act in ["id", "add1", "add2", "add3", "sub1", "mul2", "mul3"]:
                    lp, lvl = prior.log_prob(slot, act)
                    backoffs[lvl] += 1
                    new_score = score + lp - lambda_val
                    new_prims = prims + [act]
                    t_key = (depth, len(new_prims), tuple(new_prims))
                    heapq.heappush(heap, (-new_score, t_key, new_prims, rem_slots, depth))

            elif slot_type == "Predicate":
                for act in ["eq0", "gt0", "ge0", "lt0", "even", "odd", "mod3_0"]:
                    lp, lvl = prior.log_prob(slot, act)
                    backoffs[lvl] += 1
                    new_score = score + lp - lambda_val
                    new_prims = prims + [act]
                    t_key = (depth, len(new_prims), tuple(new_prims))
                    heapq.heappush(heap, (-new_score, t_key, new_prims, rem_slots, depth))

            elif slot_type == "ListExpr":
                # Option 1: Close list with xs
                if depth >= 1:
                    ast_candidate = reconstruct_ast(prims)
                    c_key = ast_candidate.canonical_key()
                    heapq.heappush(heap, (-score, c_key, prims, rem_slots, depth))

                # Option 2: Expand with map or filter if depth < 3
                if depth < 3:
                    for act in ["map", "filter"]:
                        lp, lvl = prior.log_prob(slot, act)
                        backoffs[lvl] += 1
                        new_score = score + lp - lambda_val
                        new_slots = [("map", 0, "ScalarOp"), ("map", 1, "ListExpr")] + rem_slots if act == "map" else [("filter", 0, "Predicate"), ("filter", 1, "ListExpr")] + rem_slots
                        new_prims = prims + [act]
                        t_key = (depth + 1, len(new_prims), tuple(new_prims))
                        heapq.heappush(heap, (-new_score, t_key, new_prims, new_slots, depth + 1))

    if expansions >= max_expansions:
        hit_expansion_cap = True

    return {
        "task": task_name,
        "solved": solved,
        "executions": executions if solved else budget_k,
        "expansions": expansions,
        "solution": solution_render,
        "backoffs": dict(backoffs),
        "hit_expansion_cap": hit_expansion_cap
    }


# =====================================================================
# 6. Main Pilot Runner & Output Reporting
# =====================================================================

def main():
    print("=" * 76)
    print("Path A Arm 2 Pilot: Count-Based Prior with Best-First Search (beta = 0)")
    print("Execution Budget Cap: K = 500 per task | Parsimony Penalty lambda = 0.05")
    print("=" * 76)

    # 1. Extract Training Transitions from 10 Calibration Tasks
    all_programs = enumerate_space(max_depth=3)
    ranks = audit_task_ranks(all_programs)
    calib_task_names = [
        ("T01_sum", "IntExpr"), ("T02_count", "IntExpr"), ("T03_double_all", "ListExpr"),
        ("T04_add_one", "ListExpr"), ("T05_filter_positive", "ListExpr"), ("T06_filter_evens", "ListExpr"),
        ("T07_count_positive", "IntExpr"), ("T08_sum_doubled", "IntExpr"), ("T09_add_two_all", "ListExpr"),
        ("T10_filter_ge0", "ListExpr")
    ]

    training_transitions = []
    for t_name, r_type in calib_task_names:
        p = all_programs[ranks[t_name]["syntactic_rank"] - 1]
        trans = get_canonical_transitions(p, r_type)
        training_transitions.extend(trans)

    assert len(training_transitions) == 22, f"Expected 22 transitions, got {len(training_transitions)}"
    print(f"Loaded exactly {len(training_transitions)} training transitions from T01-T10 solutions.")
    print("-" * 76)

    # 2. Build Count-Based Prior
    prior = CountBasedPrior(training_transitions, alpha=1.0)

    # 3. Gate 1 Diagnostic Evaluation
    print("Running Gate 1 Diagnostic (50 Synthetic ASTs, 239 Transitions)...")
    gate1 = evaluate_gate1_diagnostic(prior)
    print(f"Gate 1 Diagnostic Top-1 Accuracy: {gate1['overall_correct']}/{gate1['overall_total']} "
          f"({gate1['overall_acc']*100:.1f}%) vs Chance ({gate1['overall_chance']*100:.1f}%)")
    for d, d_data in gate1["depth_stats"].items():
        d_acc = d_data["correct"] / d_data["total"] * 100
        d_ch = d_data["chance"] / d_data["total"] * 100
        print(f"  Depth {d}: {d_data['correct']}/{d_data['total']} ({d_acc:.1f}%) vs Chance ({d_ch:.1f}%)")
    print("-" * 76)

    # 4. Run Best-First Search on 10 Held-Out Tasks
    heldout_tasks = sorted([name for name in TASKS if name.startswith("H")])

    results = []
    total_executions = 0
    solved_count = 0

    print("Executing Arm 2 Best-First Search across 10 Held-Out Tasks:")
    for t_name in heldout_tasks:
        res = search_task(t_name, prior, budget_k=500, max_expansions=100000, lambda_val=0.05)
        results.append(res)

        status_str = "yes" if res["solved"] else "no"
        sol_display = f" | Solution: {res['solution']}" if res["solved"] else ""
        b_info = f"L1={res['backoffs'].get('level1_context', 0)}, L2={res['backoffs'].get('level2_slot', 0)}"
        print(f"Task: {res['task']:<26} | Solved: {status_str:<3} | Executions: {res['executions']:>4} | Expansions: {res['expansions']:>5} | Backoff: {b_info}{sol_display}")

        if res["solved"]:
            solved_count += 1
        total_executions += res["executions"]

    mean_executions = total_executions / len(heldout_tasks)

    # 5. Summary & Comparison with Enumeration Baseline
    print("-" * 76)
    print("Summary:")
    print(f"  Arm 2 Solved: {solved_count} / {len(heldout_tasks)} ({solved_count / len(heldout_tasks) * 100:.1f}%) [Enum Baseline: 7 / 10]")
    print(f"  Arm 2 Mean Executions: {mean_executions:.1f} (unsolved capped at 500) [Enum Baseline: 224.2]")
    print("=" * 76)
    print("Gate 1 Diagnostic Table:")
    print(f"  {'Depth':<8} | {'Correct/Total':<14} | {'Prior Top-1 Acc':<16} | {'Chance Baseline':<16}")
    print(f"  {'-'*8} | {'-'*14} | {'-'*16} | {'-'*16}")
    for d, d_data in gate1["depth_stats"].items():
        d_acc = d_data["correct"] / d_data["total"] * 100
        d_ch = d_data["chance"] / d_data["total"] * 100
        print(f"  Depth {d:<2} | {d_data['correct']:>3}/{d_data['total']:<9} | {d_acc:>13.1f}% | {d_ch:>13.1f}%")
    print(f"  {'-'*8} | {'-'*14} | {'-'*16} | {'-'*16}")
    print(f"  Overall  | {gate1['overall_correct']:>3}/{gate1['overall_total']:<9} | {gate1['overall_acc']*100:>13.1f}% | {gate1['overall_chance']*100:>13.1f}%")
    print("=" * 76)
    print("Limitations:")
    print("  1. Only 10 training programs (22 transitions).")
    print("  2. Vocabulary and held-out tasks were chosen by the same process, so generalization claims are limited to this benchmark.")
    print("  3. Evaluations were executed using in-process AST reference interpreter, not isolated subprocess sandboxes.")
    print("=" * 76)


if __name__ == "__main__":
    main()

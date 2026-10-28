"""
scripts/path_a_feasibility_check.py
Strict, deterministic feasibility and search space audit for Path A Neurosymbolic Program Synthesis.
Implements a mathematically complete Total-Order Comparator over all well-typed ASTs.
Computes both Syntactic Rank and Behavioral Deduplication Rank across all 20 tasks.
"""
from typing import Dict, Any, List, Tuple, Optional, Set
from dataclasses import dataclass


# =====================================================================
# 1. Total-Order Comparator & AST Representation
# =====================================================================

# Primitive Index Mappings (Strictly Frozen for Total Ordering)
CAT_INPUT  = 0
CAT_LEN    = 1
CAT_MAP    = 2
CAT_FILTER = 3
CAT_FOLD   = 4

SCALAR_OPS = [
    ("id",   lambda x: x,     "x",       0),
    ("add1", lambda x: x + 1, "(x + 1)", 1),
    ("add2", lambda x: x + 2, "(x + 2)", 2),
    ("add3", lambda x: x + 3, "(x + 3)", 3),
    ("sub1", lambda x: x - 1, "(x - 1)", 4),
    ("mul2", lambda x: x * 2, "(x * 2)", 5),
    ("mul3", lambda x: x * 3, "(x * 3)", 6),
]

PREDICATES = [
    ("eq0",    lambda x: x == 0,     "(x == 0)",            0),
    ("gt0",    lambda x: x > 0,      "(x > 0)",             1),
    ("ge0",    lambda x: x >= 0,     "(x >= 0)",            2),
    ("lt0",    lambda x: x < 0,      "(x < 0)",             3),
    ("even",   lambda x: x % 2 == 0, "(x % 2 == 0)",        4),
    ("odd",    lambda x: x % 2 != 0, "(not (x % 2 == 0))",  5),
    ("mod3_0", lambda x: x % 3 == 0, "(x % 3 == 0)",       6),
]

FOLD_REDUCERS = [
    ("add", lambda acc, x: acc + x, "acc + x", 0, 0), # (name, fn, repr, init, idx)
    ("sub", lambda acc, x: acc - x, "acc - x", 0, 1),
    ("mul", lambda acc, x: acc * x, "acc * x", 1, 2),
]


class ASTNode:
    def depth(self) -> int: raise NotImplementedError
    def size(self) -> int: raise NotImplementedError
    def render(self) -> str: raise NotImplementedError
    def eval(self, xs: List[int]) -> Any: raise NotImplementedError
    def canonical_key(self) -> Tuple: raise NotImplementedError


@dataclass(frozen=True)
class InputList(ASTNode):
    def depth(self) -> int: return 0
    def size(self) -> int: return 1
    def render(self) -> str: return "xs"
    def eval(self, xs: List[int]) -> Any: return list(xs)
    def canonical_key(self) -> Tuple:
        return (0, 1, CAT_INPUT, 0, ())


@dataclass(frozen=True)
class LenNode(ASTNode):
    child: ASTNode

    def depth(self) -> int: return 1 + self.child.depth()
    def size(self) -> int: return 1 + self.child.size()
    def render(self) -> str: return f"len({self.child.render()})"
    def eval(self, xs: List[int]) -> Any:
        sub = self.child.eval(xs)
        if sub is None: return None
        return len(sub)
    def canonical_key(self) -> Tuple:
        return (self.depth(), self.size(), CAT_LEN, 0, self.child.canonical_key())


@dataclass(frozen=True)
class MapNode(ASTNode):
    op_name: str
    func: Any
    op_repr: str
    op_idx: int
    child: ASTNode

    def depth(self) -> int: return 1 + self.child.depth()
    def size(self) -> int: return 2 + self.child.size()
    def render(self) -> str: return f"map(lambda x: {self.op_repr}, {self.child.render()})"
    def eval(self, xs: List[int]) -> Any:
        sub = self.child.eval(xs)
        if sub is None: return None
        try:
            return [self.func(x) for x in sub]
        except Exception:
            return None
    def canonical_key(self) -> Tuple:
        return (self.depth(), self.size(), CAT_MAP, self.op_idx, self.child.canonical_key())


@dataclass(frozen=True)
class FilterNode(ASTNode):
    pred_name: str
    pred: Any
    pred_repr: str
    pred_idx: int
    child: ASTNode

    def depth(self) -> int: return 1 + self.child.depth()
    def size(self) -> int: return 2 + self.child.size()
    def render(self) -> str: return f"filter(lambda x: {self.pred_repr}, {self.child.render()})"
    def eval(self, xs: List[int]) -> Any:
        sub = self.child.eval(xs)
        if sub is None: return None
        try:
            return [x for x in sub if self.pred(x)]
        except Exception:
            return None
    def canonical_key(self) -> Tuple:
        return (self.depth(), self.size(), CAT_FILTER, self.pred_idx, self.child.canonical_key())


@dataclass(frozen=True)
class FoldNode(ASTNode):
    reducer_name: str
    reducer: Any
    reducer_repr: str
    init: int
    red_idx: int
    child: ASTNode

    def depth(self) -> int: return 1 + self.child.depth()
    def size(self) -> int: return 3 + self.child.size()
    def render(self) -> str: return f"fold({self.init}, lambda acc, x: {self.reducer_repr}, {self.child.render()})"
    def eval(self, xs: List[int]) -> Any:
        sub = self.child.eval(xs)
        if sub is None: return None
        acc = self.init
        try:
            for x in sub:
                acc = self.reducer(acc, x)
            return acc
        except Exception:
            return None
    def canonical_key(self) -> Tuple:
        return (self.depth(), self.size(), CAT_FOLD, self.red_idx, self.child.canonical_key())


# =====================================================================
# 2. Total-Order Space Generation
# =====================================================================

def enumerate_space(max_depth: int = 3) -> List[ASTNode]:
    """
    Generates all well-typed programs up to max_depth and sorts them
    strictly by the Total-Order Comparator key canonical_key().
    """
    levels_list: Dict[int, List[ASTNode]] = {0: [InputList()]}
    all_programs: List[ASTNode] = [InputList()]

    for d in range(1, max_depth + 1):
        cur_lists = []
        cur_progs = []

        # 1. Map over prev lists of depth d-1
        for op_name, fn, rpr, op_idx in SCALAR_OPS:
            for l in levels_list[d - 1]:
                m = MapNode(op_name, fn, rpr, op_idx, l)
                cur_lists.append(m)
                cur_progs.append(m)

        # 2. Filter over prev lists of depth d-1
        for pred_name, pr, rpr, pr_idx in PREDICATES:
            for l in levels_list[d - 1]:
                f = FilterNode(pred_name, pr, rpr, pr_idx, l)
                cur_lists.append(f)
                cur_progs.append(f)

        # 3. Len over prev lists of depth d-1
        for l in levels_list[d - 1]:
            cur_progs.append(LenNode(l))

        # 4. Fold over prev lists of depth d-1
        for red_name, fn, rpr, init, red_idx in FOLD_REDUCERS:
            for l in levels_list[d - 1]:
                cur_progs.append(FoldNode(red_name, fn, rpr, init, red_idx, l))

        levels_list[d] = cur_lists
        all_programs.extend(cur_progs)

    # Sort strictly by Total-Order canonical_key()
    all_programs.sort(key=lambda node: node.canonical_key())
    return all_programs


# =====================================================================
# 3. 20-Task Family Specification
# =====================================================================

TASKS = {
    # Calibration / Training Tasks
    "T01_sum": [([1, 2, 3, 4], 10), ([-3, 0, 2, 5, -1], 3), ([-2, -4, -6], -12), ([7], 7), ([], 0)],
    "T02_count": [([1, 2, 3, 4], 4), ([-3, 0, 2, 5, -1], 5), ([-2, -4, -6], 3), ([7], 1), ([], 0)],
    "T03_double_all": [([1, 2, 3, 4], [2, 4, 6, 8]), ([-3, 0, 2], [-6, 0, 4]), ([-2, -4], [-4, -8]), ([7], [14]), ([], [])],
    "T04_add_one": [([1, 2, 3], [2, 3, 4]), ([-3, 0, 2], [-2, 1, 3]), ([-2, -4], [-1, -3]), ([7], [8]), ([], [])],
    "T05_filter_positive": [([1, -2, 3, -4], [1, 3]), ([-3, 0, 2], [2]), ([-2, -4], []), ([7], [7]), ([], [])],
    "T06_filter_evens": [([1, 2, 3, 4], [2, 4]), ([-3, 0, 2, 5], [0, 2]), ([-2, -4], [-2, -4]), ([7], []), ([], [])],
    "T07_count_positive": [([1, -2, 3, -4], 2), ([-3, 0, 2], 1), ([-2, -4], 0), ([7], 1), ([], 0)],
    "T08_sum_doubled": [([1, 2, 3, 4], 20), ([-3, 0, 2, 5, -1], 6), ([-2, -4, -6], -24), ([7], 14), ([], 0)],
    "T09_add_two_all": [([1, 2, 3], [3, 4, 5]), ([-3, 0, 2], [-1, 2, 4]), ([-2, -4], [0, -2]), ([7], [9]), ([], [])],
    "T10_filter_ge0": [([1, -2, 0, 3], [1, 0, 3]), ([-3, 0, 2], [0, 2]), ([-2, -4], []), ([7], [7]), ([], [])],

    # Held-Out Evaluation Tasks
    "H01_filter_odds": [([1, 2, 3, 4], [1, 3]), ([-3, 0, 2, 5], [-3, 5]), ([-2, -4], []), ([7], [7]), ([], [])],
    "H02_double_evens": [([1, 2, 3, 4], [4, 8]), ([-3, 0, 2, 5], [0, 4]), ([-2, -4], [-4, -8]), ([7], []), ([], [])],
    "H03_add_two_positives": [([1, -2, 3], [3, 5]), ([-3, 0, 2], [4]), ([-2, -4], []), ([7], [9]), ([], [])],
    "H04_count_evens": [([1, 2, 3, 4], 2), ([-3, 0, 2, 5], 2), ([-2, -4, -6], 3), ([7], 0), ([], 0)],
    "H05_count_odds": [([1, 2, 3, 4], 2), ([-3, 0, 2, 5], 2), ([-2, -4], 0), ([7], 1), ([], 0)],
    "H06_sum_evens": [([1, 2, 3, 4], 6), ([-3, 0, 2, 5], 2), ([-2, -4, -6], -12), ([7], 0), ([], 0)],
    "H07_sum_positives": [([1, -2, 3, -4], 4), ([-3, 0, 2, 5, -1], 7), ([-2, -4], 0), ([7], 7), ([], 0)],
    "H08_double_positive_evens": [([1, -2, 4, -6, 3], [8]), ([-3, 0, 2, 5], [4]), ([-2, -4], []), ([6], [12]), ([], [])],
    "H09_sum_doubled_evens": [([1, 2, 3, 4], 12), ([-3, 0, 2, 5], 4), ([-2, -4, -6], -24), ([7], 0), ([], 0)],
    "H10_sum_add1_evens": [([1, 2, 3, 4], 8), ([-3, 0, 2, 5], 4), ([-2, -4, -6], -9), ([7], 0), ([], 0)]
}


def audit_task_ranks(all_programs: List[ASTNode]):
    """
    Evaluates both Syntactic Rank and Behavioral Deduplication Rank for all 20 tasks.
    """
    results = {}
    
    # Pre-evaluate output tuples for behavioral deduplication per task
    for task_name, test_cases in TASKS.items():
        inputs = [inp for inp, _ in test_cases]
        expected_tuple = tuple(exp if not isinstance(exp, list) else tuple(exp) for _, exp in test_cases)
        
        seen_behaviors: Set[Tuple] = set()
        first_syntactic_rank = None
        first_dedup_rank = None
        best_prog = None
        executions_executed = 0

        for rank, prog in enumerate(all_programs, start=1):
            # Compute behavior on the 5 inputs
            out_list = []
            valid_eval = True
            for inp in inputs:
                val = prog.eval(inp)
                if val is None:
                    valid_eval = False
                    break
                out_list.append(tuple(val) if isinstance(val, list) else val)
            
            behavior_tuple = tuple(out_list) if valid_eval else None
            
            # Check if this behavior was already seen
            is_new_behavior = False
            if behavior_tuple is not None:
                if behavior_tuple not in seen_behaviors:
                    seen_behaviors.add(behavior_tuple)
                    is_new_behavior = True
                    executions_executed += 1
            else:
                # Crashes/invalid evaluations count as 1 execution attempt
                executions_executed += 1

            # Check if this program solves the task
            if behavior_tuple == expected_tuple:
                if first_syntactic_rank is None:
                    first_syntactic_rank = rank
                    first_dedup_rank = executions_executed
                    best_prog = prog
                    break

        results[task_name] = {
            "syntactic_rank": first_syntactic_rank,
            "dedup_rank": first_dedup_rank,
            "depth": best_prog.depth() if best_prog else None,
            "size": best_prog.size() if best_prog else None,
            "solution": best_prog.render() if best_prog else "UNREACHABLE"
        }

    return results


def main():
    print("====================================================================================================")
    print("  PATH A FEASIBILITY AUDIT: TOTAL-ORDER COMPARATOR & DUAL RANK ANALYSIS")
    print("====================================================================================================")
    
    all_programs = enumerate_space(max_depth=3)
    
    d0_count = sum(1 for p in all_programs if p.depth() == 0)
    d1_count = sum(1 for p in all_programs if p.depth() == 1)
    d2_count = sum(1 for p in all_programs if p.depth() == 2)
    d3_count = sum(1 for p in all_programs if p.depth() == 3)
    
    print(f"Programs at Depth 0:                {d0_count:<5} (Cumulative: {d0_count})")
    print(f"Programs at Depth 1:                {d1_count:<5} (Cumulative: {d0_count + d1_count})")
    print(f"Programs at Depth 2:                {d2_count:<5} (Cumulative: {d0_count + d1_count + d2_count})")
    print(f"Programs at Depth 3:                {d3_count:<5} (Cumulative: {len(all_programs)})")
    print(f"\nTotal Syntactic Search Space at D <= 3: {len(all_programs)} programs.")
    print("Total Order Comparator: (depth, size, category_idx, primitive_idx, child_key)")

    ranks = audit_task_ranks(all_programs)
    
    print("\n----------------------------------------------------------------------------------------------------")
    print(f"{'Task Name':<28} | {'Depth':<5} | {'Syn Rank':<8} | {'Dedup Rank':<10} | Canonical AST Solution")
    print("----------------------------------------------------------------------------------------------------")
    
    for t_name, r in ranks.items():
        syn_str = str(r["syntactic_rank"]) if r["syntactic_rank"] else "> 3799"
        dedup_str = str(r["dedup_rank"]) if r["dedup_rank"] else "> 3799"
        depth_str = str(r["depth"]) if r["depth"] is not None else "-"
        print(f"{t_name:<28} | {depth_str:<5} | {syn_str:<8} | {dedup_str:<10} | {r['solution']}")

    heldout_syn = [ranks[k]["syntactic_rank"] for k in ranks if k.startswith("H")]
    heldout_dedup = [ranks[k]["dedup_rank"] for k in ranks if k.startswith("H")]
    
    print("\n====================================================================================================")
    print("HELD-OUT TASKS SUMMARY (N = 10):")
    print(f"  Syntactic Ranks:         {sorted(heldout_syn)}")
    print(f"  Dedup Execution Ranks:   {sorted(heldout_dedup)}")
    print(f"  Solved at K = 500 under Syntactic Order: {sum(1 for r in heldout_syn if r <= 500)}/10")
    print(f"  Solved at K = 500 under Dedup Order:     {sum(1 for r in heldout_dedup if r <= 500)}/10")
    print("====================================================================================================")


if __name__ == "__main__":
    main()

"""
scripts/enum_pilot.py
Uniform canonical enumeration pilot for Path A Neurosymbolic Synthesis.
Evaluates the 10 held-out tasks (H01 to H10) up to budget K = 500 executions.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import List, Tuple, Any
from scripts.path_a_feasibility_check import enumerate_space, audit_task_ranks, TASKS, ASTNode


def evaluate_program_on_visible_cases(prog: ASTNode, test_cases: List[Tuple[List[int], Any]]) -> bool:
    """Evaluates prog on all test cases; returns True iff all pass."""
    for inp, expected in test_cases:
        try:
            actual = prog.eval(inp)
            if actual != expected:
                return False
        except Exception:
            return False
    return True


def run_enumeration_pilot(budget_k: int = 500):
    print("=" * 70)
    print("Path A Uniform Canonical Enumeration Pilot (10 Held-Out Tasks)")
    print(f"Execution Budget Cap: K = {budget_k} per task")
    print("=" * 70)

    # 1. Generate canonical search space (sorted strictly by canonical_key)
    all_programs = enumerate_space(max_depth=3)
    print(f"Total well-typed candidate programs in search space: {len(all_programs)}")
    print("-" * 70)

    # 2. Select held-out tasks (H01 through H10)
    heldout_tasks = sorted([name for name in TASKS if name.startswith("H")])

    results = []
    total_executions = 0
    solved_count = 0

    for task_name in heldout_tasks:
        test_cases = TASKS[task_name]
        solved = False
        executions_used = 0
        solution_render = ""

        # Walk canonical order from index 1
        for i, prog in enumerate(all_programs, start=1):
            if i > budget_k:
                break
            executions_used = i
            if evaluate_program_on_visible_cases(prog, test_cases):
                solved = True
                solution_render = prog.render()
                break

        if not solved:
            executions_used = budget_k

        results.append({
            "task": task_name,
            "solved": solved,
            "executions": executions_used,
            "solution": solution_render
        })

        status_str = "yes" if solved else "no"
        sol_display = f" | Solution: {solution_render}" if solved else ""
        print(f"Task: {task_name:<26} | Solved: {status_str:<3} | Executions: {executions_used:>4}{sol_display}")

        if solved:
            solved_count += 1
        total_executions += executions_used

    mean_executions = total_executions / len(heldout_tasks)

    print("-" * 70)
    print("Summary:")
    print(f"  Solved: {solved_count} / {len(heldout_tasks)} ({solved_count / len(heldout_tasks) * 100:.1f}%)")
    print(f"  Mean Executions: {mean_executions:.1f} (unsolved capped at {budget_k})")
    print("=" * 70)

    return results, solved_count, mean_executions


if __name__ == "__main__":
    run_enumeration_pilot(budget_k=500)

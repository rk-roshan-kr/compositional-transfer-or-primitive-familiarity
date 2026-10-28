"""
compositional_transfer
======================
Empirical decomposition and causal isolation suite for prior-guided program synthesis.
Investigating whether statistical prior guidance achieves compositional generalization
or exploits single-primitive familiarity biases in typed functional domain-specific languages.
"""

from scripts.path_a_feasibility_check import (
    ASTNode,
    InputList,
    LenNode,
    MapNode,
    FilterNode,
    FoldNode,
    SCALAR_OPS,
    PREDICATES,
    FOLD_REDUCERS,
    enumerate_space,
    audit_task_ranks,
    TASKS,
)

__version__ = "1.0.0"
__author__ = "Roshan Kumar"
__all__ = [
    "ASTNode",
    "InputList",
    "LenNode",
    "MapNode",
    "FilterNode",
    "FoldNode",
    "SCALAR_OPS",
    "PREDICATES",
    "FOLD_REDUCERS",
    "enumerate_space",
    "audit_task_ranks",
    "TASKS",
]

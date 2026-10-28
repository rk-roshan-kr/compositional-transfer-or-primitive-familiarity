# Design Specification: Path A Neurosymbolic Program Synthesis (A Compositional Transfer Pilot)
**File:** `docs/design/path_a_neurosymbolic_synthesis.md`  
**Date:** 2026-10-26  
**Status:** PROPOSED BENCHMARK SPECIFICATION (Under Supervisor Review)  
**Role:** Research Agent  
**Audience:** Research Supervisor / Scientific Auditor  

---

## 1. Core Benchmark Question & Study Framing

### 1.1. Core Benchmark Question
> **Does a learned neural prior and continuous failure-attractor memory find Depth-3 program compositions faster than canonical uniform enumeration in a 3,799-program syntactic space under a fixed budget of 500 program executions?**

### 1.2. Primary Metric: Syntactic Rank ($R_{\text{syn}}$)
The primary evaluation metric of this benchmark is strictly **Syntactic Rank ($R_{\text{syn}}$)**:
- Syntactic Rank measures the exact program index in the frozen canonical enumeration sequence at which a passing program is found.
- It requires **zero oracle assumptions**, has **zero interpreter-overhead ambiguity**, and provides a wide, robust margin for testing: all three Depth-3 held-out tasks (H08, H09, H10) sit at syntactic ranks **1571, 3293, and 3237**—more than 1,000 evaluations beyond the $K = 500$ budget under the pinned canonical order.
- **Behavioral Deduplication Rank ($R_{\text{dedup}}$)** is reported strictly as a secondary sensitivity analysis.

### 1.3. Descriptive Pilot Framing ($N = 10$)
Because the evaluation set consists of $N = 10$ held-out tasks, **no inferential hypothesis testing, p-values, or bootstrap separation claims are made**. Bootstrap intervals on 10 paired observations where 7 tasks are solved early by both arms are statistically uninformative. 

The study is formally designed and reported as an **exploratory descriptive pilot** reporting:
1. Exact per-task program execution counts ($E_i \in [1, 500]$) across all 10 tasks for all arms.
2. Total tasks solved ($N_{\text{solved}} \in [0, 10]$).
3. Sample efficiency ratio ($\bar{E}_{\text{Enum}} / \bar{E}_{\text{Earthos}}$).
4. Secondary metrics: total prefix expansions ($X \in \mathbb{N}$), wall-clock search time ($T \in \mathbb{R}$ seconds), and generalization on hidden inputs.

---

## 2. The Exact Domain-Specific Language (DSL) & Semantics

### 2.1. AST Depth Definition
Depth is defined strictly by expression tree level:
- **Depth 0 (Leaves):** The input list variable `xs`.
- **Depth 1:** A single combinator applied to `xs`: `map(f, xs)`, `filter(p, xs)`, `fold(init, f, xs)`, or `len(xs)`.
- **Depth $d + 1$:** A combinator applied to a list sub-expression of depth $d$:
  $$\text{depth}(Node) = 1 + \text{depth}(\text{child\_list\_expr})$$
- **Depth Limit:** Strictly **$D \le 3$**.

### 2.2. Generic Primitive Families ($\mathcal{P}$)
The DSL contains zero task-specific conveniences (no `filter_even`, no `accumulate`). It consists of exactly **21 generic typed primitives** (4 combinators + 7 scalar operations + 7 predicates + 3 reducers; `xs` serves as the base input list terminal).

*(Note on 18 vs. 21: The count 18 is the number of distinct complete Depth-1 programs formed over `xs` [7 `map` + 7 `filter` + 3 `fold` + 1 `len` = 18], whereas the action vocabulary consists of the 21 primitives listed below).*

1. **Higher-Order Combinators (4):**
   - `map(f, xs)`: Applies scalar function $f: \text{Int} \to \text{Int}$ to each element.
   - `filter(p, xs)`: Retains elements satisfying predicate $p: \text{Int} \to \text{Bool}$.
   - `fold(init, f, xs)`: Left fold reducing list with binary operator $f: (\text{Int}, \text{Int}) \to \text{Int}$.
   - `len(xs)`: Returns integer length of list.

2. **Generic Scalar Operations ($f: \text{Int} \to \text{Int}$):**
   - Identity: `id` ($x \mapsto x$). *(Retained as standard functional identity; semantic redundancies like `map(id, xs)` are standard properties of syntactic spaces).*
   - Affine transformations with constants $\mathcal{C} = \{1, 2, 3\}$:
     `add1` ($x+1$), `add2` ($x+2$), `add3` ($x+3$), `sub1` ($x-1$), `mul2` ($x \cdot 2$), `mul3` ($x \cdot 3$).

3. **Generic Predicates ($p: \text{Int} \to \text{Bool}$):**
   - Comparison to zero: `eq0` ($x == 0$), `gt0` ($x > 0$), `ge0` ($x \ge 0$), `lt0` ($x < 0$).
   - Modular divisibility: `even` ($x \% 2 == 0$), `odd` ($\text{not}(x \% 2 == 0)$), `mod3_0` ($x \% 3 == 0$).

4. **Fold Reducers ($f: (\text{Int}, \text{Int}) \to \text{Int}$ with pinned init):**
   - `add`: $\text{acc} + x$ with $\text{init} = 0$.
   - `sub`: $\text{acc} - x$ with $\text{init} = 0$.
   - `mul`: $\text{acc} \cdot x$ with $\text{init} = 1$.
   *(Note: `max` has been formally removed from the DSL to eliminate boundary failures on negative inputs).*

### 2.3. Pinned Signatures & Semantics
- **Pinned Fold Signature:** Strictly $\text{fold}(\text{init}, \text{reducer}, \text{xs})$.
- **Modulo Semantics:** Python `%` semantics pinned: `a % b` has the sign of $b$.
- **Bottom Value ($\bot$):** Division by zero yields typed failure $\bot$, which immediately causes the candidate program to fail the test case.
- **Syntactic Search Space Size:** Formally verified via `scripts/path_a_feasibility_check.py` and unit tested in `tests/test_path_a_feasibility.py`:
  - Depth 0: 1 program
  - Depth 1: 18 programs (Cumulative: 19)
  - Depth 2: 252 programs (Cumulative: 271)
  - Depth 3: 3,528 programs (Cumulative: 3,799)
  - **Total Syntactic Search Space:** Exactly **3,799 well-typed ASTs**.

---

## 3. Total-Order Comparator & Canonical Enumeration Order

To eliminate arbitrary rank fluctuations, tie-breaking ambiguities, and informal sorting labels, Uniform Enumerative Search evaluates candidate programs in a mathematically complete **total order** defined by the comparator key `canonical_key(node)`.

### 3.1. Formal Total-Order Comparator Code Specification
```python
from typing import Tuple

# Fixed Category Indices (Discrete Serialization Tag)
CAT_INPUT  = 0  # Terminal variable xs
CAT_LEN    = 1  # len(xs)
CAT_MAP    = 2  # map(fn, xs)
CAT_FILTER = 3  # filter(pred, xs)
CAT_FOLD   = 4  # fold(init, red, xs)

# Fixed Primitive Indices
SCALAR_OPS = {
    "id": 0, "add1": 1, "add2": 2, "add3": 3,
    "sub1": 4, "mul2": 5, "mul3": 6
}
PREDICATES = {
    "eq0": 0, "gt0": 1, "ge0": 2, "lt0": 3,
    "even": 4, "odd": 5, "mod3_0": 6
}
FOLD_REDUCERS = {
    "add": 0,  # init = 0
    "sub": 1,  # init = 0
    "mul": 2,  # init = 1
}

def canonical_key(node: ASTNode) -> Tuple:
    """
    Returns a strict lexicographical tuple (depth, size, category_idx, primitive_idx, child_key).
    Because all tuple elements at every level are integers with unique indices and all recursion
    terminates at InputList with (), this defines a strict total order with zero ties.
    """
    if isinstance(node, InputList):
        return (0, 1, CAT_INPUT, 0, ())
    elif isinstance(node, LenNode):
        return (node.depth(), node.size(), CAT_LEN, 0, canonical_key(node.child))
    elif isinstance(node, MapNode):
        return (node.depth(), node.size(), CAT_MAP, node.op_idx, canonical_key(node.child))
    elif isinstance(node, FilterNode):
        return (node.depth(), node.size(), CAT_FILTER, node.pred_idx, canonical_key(node.child))
    elif isinstance(node, FoldNode):
        return (node.depth(), node.size(), CAT_FOLD, node.red_idx, canonical_key(node.child))
    raise TypeError(f"Unknown AST node type: {type(node)}")
```

### 3.2. Total-Order Guarantees
1. **Reflexivity, Antisymmetry, Transitivity:** Guaranteed by standard lexicographical tuple comparison over discrete non-negative integer fields.
2. **Totality:** For any two well-typed ASTs $P_1, P_2 \in \mathcal{P}$, either $\text{canonical\_key}(P_1) < \text{canonical\_key}(P_2)$, $\text{canonical\_key}(P_1) > \text{canonical\_key}(P_2)$, or $P_1$ is syntactically identical to $P_2$. Injective uniqueness is asserted by test suite across all 3,799 programs.
3. **No Informal Labels:** Primitive ordering is governed strictly by the frozen integer indices above.

### 3.3. Behavioral Deduplication Sensitivity Analysis & Accounting
- **Primary Metric Invariance Across All Arms:** The primary benchmark metric strictly counts every candidate program execution in the isolated subprocess sandbox across all three arms ($E_i \in \mathbb{N}$). Behavioral deduplication applies strictly to the secondary sensitivity analysis ($R_{\text{dedup}}$) and does not alter primary metric accounting for any arm.
- **Cache Definition:** When candidate program $P$ is evaluated on the 5 visible test cases $\mathcal{E}$, its output tuple $B(P) = (y_1, \dots, y_5)$ is recorded in a hash table $\mathcal{B}$.
- **Duplicate Skipping:** If a subsequent candidate $P'$ produces an output tuple already in $\mathcal{B}$, $P'$ is recorded as a semantic duplicate.
- **Interpreter-Cost Assumption:** Dedup assumes the existence of a fast in-process AST reference interpreter whose compute cost (~microseconds) is treated as negligible compared to isolated subprocess sandbox execution. Under this assumption, semantic duplicates are skipped before sandbox dispatch and do not consume sandbox execution budget. If reference interpreter evaluations are instead charged to budget, dedup execution counts are identical to syntactic evaluations.
- **Memory Interaction:** Duplicate candidates **do not generate failure traces** in memory $\mathcal{M}$. The attractor basin for that failure mode is already established by the initial program that generated $B(P)$, preventing memory dilution with redundant copies of identical semantic failures.

---

## 4. Search Algorithm: Learned Prior Search with Attractor Repulsion (LPS-AR)

Arm 3 runs **Learned Prior Search with Attractor Repulsion (LPS-AR)** over partial program prefixes via best-first queue discipline:

```
Algorithm 1: Learned Prior Search with Attractor Repulsion (LPS-AR)
Input: Test cases E = {(x_k, y_k)}, Execution Budget K = 500, Weight beta = 1.0, Parsimony lambda = 0.05
Output: Passing program P* or FAIL

1: Initialize Priority Queue Q with root partial program prefix pi_0 = [?] (S_0(pi_0) = 0.0)
2: Initialize Stored Failure Memory M = []
3: executions_count = 0
4: 
5: while Q is not empty and executions_count < K do
6:     Pop candidate prefix pi from Q with highest priority S_eff(pi)
7:     
8:     if pi is a complete executable program then
9:         executions_count += 1
10:        Execute pi on all test cases E in isolated sandbox, capturing pass rate and actual scalar reward r
11:        if pi passes 100% of test cases then
12:            return pi (SUCCESS)
13:        else
14:            Extract pre-execution query vector z_q = StateEncoder(pi, E)
15:            Store failure trace (z_q, a_root = pi.root_combinator, r = r, severity = 10.0) in M
16:        end if
17:    else
18:        // pi is a partial program with an open slot
19:        valid_actions = GetTypeValidCompletions(pi)
20:        for each a in valid_actions do
21:            child_pi = Append(pi, a)
22:            
23:            // Accumulate Learned Prior Score S_0 with per-step parsimony penalty lambda
24:            S_0(child_pi) = S_0(pi) + ln P_theta(a | pi, E) - lambda
25:            
26:            // Evaluate Normalized Continuous Kernel Resonance against Failure Memory M
27:            z_cand = StateEncoder(child_pi, E)
28:            R = ComputeKernelResonance(z_cand, M)
29:            
30:            // Soft Attractor Repulsion (Down-ranking, not hard deletion)
31:            S_eff(child_pi) = S_0(child_pi) - beta * R
32:            
33:            Push child_pi into Q with priority S_eff(child_pi)
34:        end for
35:    end if
36: end while
37: return FAIL (Budget Exceeded)
```

### 4.1. Queue Discipline & Scoring
- **Discipline:** Best-First Search over program trees ordered by $S_{\text{eff}}(\pi)$.
- **Prefix Prior Score ($S_0(\pi)$):**
  $$S_0(\pi) = \sum_{j=1}^{|\pi|} \left[ \ln P_\theta(a_j \mid \pi_{<j}, \mathcal{E}) - \lambda \right]$$
  where $\lambda = 0.05$ is subtracted once per primitive addition. Root prefix starts at $S_0(\pi_0) = 0.0$.
- **Arm 2 vs. Arm 3 Memory Timing:** Memory traces are stored *only* after a complete program fails in the sandbox. Therefore, $\beta R \equiv 0$ until the first complete program execution fails. From that point onward, memory repels prefix expansions structurally similar to previous failures. The comparison between Arm 2 ($\beta = 0.0$) and Arm 3 ($\beta = 1.0$) isolates the causal effect of these post-failure repulsion traces.

---

## 5. Neural Prior & State Representation Architecture

### 5.1. Input Spec Feature Vector ($\mathbf{e}_{\text{spec}}(\mathcal{E}) \in \mathbb{R}^{16}$)
Computed deterministically from the 5 visible input-output pairs $\mathcal{E} = \{(x_i, y_i)\}_{i=1}^5$:
1. `mean_input_len`: $\frac{1}{5} \sum |x_i|$
2. `max_input_len`: $\max_i |x_i|$
3. `is_output_scalar`: $1.0$ if outputs are integers, $0.0$ if lists
4. `mean_output_len`: $\frac{1}{5} \sum (|y_i| \text{ if list else } 1)$
5. `len_ratio`: $\text{mean}(|y_i| / \max(|x_i|, 1))$
6. `min_input_val`: $\min_{i, x \in x_i} x / 10.0$
7. `max_input_val`: $\max_{i, x \in x_i} x / 10.0$
8. `mean_input_val`: $\text{mean}_{i, x \in x_i} x / 10.0$
9. `frac_pos`: fraction of input elements $> 0$
10. `frac_neg`: fraction of input elements $< 0$
11. `frac_zero`: fraction of input elements $== 0$
12. `frac_even`: fraction of input elements with $x \% 2 == 0$
13. `frac_odd`: fraction of input elements with $x \% 2 \ne 0$
14. `min_output_val`: $\min y / 10.0$ (or elements if list)
15. `max_output_val`: $\max y / 10.0$
16. `is_monotonic_order`: $1.0$ if list elements preserve relative input order, $0.0$ otherwise.
Standardized via LayerNorm to produce $\mathbf{e}_{\text{spec}}(\mathcal{E}) \in \mathbb{R}^{16}$.

### 5.2. AST Prefix Feature Vector ($\mathbf{e}_{\text{AST}}(\pi) \in \mathbb{R}^{16}$)
Computed deterministically from the partial program prefix $\pi$:
1. `depth`: current tree depth $d \in \{0, 1, 2, 3\}$
2. `size`: current node count $|\pi| \in \{1, \dots, 7\}$
3–7. `expected_child_type`: one-hot over 5 semantic types (List, Int, Predicate, ScalarOp, Reducer)
8–12. `combinator_counts`: bag-of-combinators count (`map`, `filter`, `fold`, `len`, `xs`)
13–16. `parent_combinator`: one-hot over parent combinator (`map`, `filter`, `fold`, `len`).
Standardized via LayerNorm to produce $\mathbf{e}_{\text{AST}}(\pi) \in \mathbb{R}^{16}$.

### 5.3. Pre-Execution Query Vector ($\mathbf{z}_q(\pi) \in \mathbb{R}^{32}$)
$$\mathbf{z}_q(\pi) = \text{LayerNorm}\left( W_{\text{ast}} \cdot \mathbf{e}_{\text{AST}}(\pi) + W_{\text{spec}} \cdot \mathbf{e}_{\text{spec}}(\mathcal{E}) \right) \in \mathbb{R}^{32}$$
Evaluated *before* program $\pi$ is ever executed. **No test results or rewards are present in $\mathbf{z}_q$.**

### 5.4. Neural Prior Network Specification
- **Architecture:** 2-layer MLP:
  - Input: $[\mathbf{e}_{\text{AST}}(\pi) \,\|\, \mathbf{e}_{\text{spec}}(\mathcal{E})] \in \mathbb{R}^{32}$
  - Hidden Layer 1: $\text{Linear}(32 \to 64) \to \text{LayerNorm}(64) \to \text{GELU}$
  - Hidden Layer 2: $\text{Linear}(64 \to 64) \to \text{LayerNorm}(64) \to \text{GELU}$
  - Output Head: $\text{Linear}(64 \to 21)$ producing logits over the 21 DSL action primitives.
  - Type Masking: All primitives not in $\mathcal{A}_{\text{valid}}(\pi)$ are masked to $-\infty$ prior to softmax.
- **Training Target & Canonical Traversal Transitions:**
  Multi-class cross-entropy over next-primitive $a_t^* \in \mathcal{A}$ in canonical solution AST prefix traversal across the 10 calibration tasks (exactly **22 total transitions**):
  - `T01_sum`: `['fold', 'add']` (2 transitions)
  - `T02_count`: `['len']` (1 transition)
  - `T03_double_all`: `['map', 'mul2']` (2 transitions)
  - `T04_add_one`: `['map', 'add1']` (2 transitions)
  - `T05_filter_positive`: `['filter', 'gt0']` (2 transitions)
  - `T06_filter_evens`: `['filter', 'even']` (2 transitions)
  - `T07_count_positive`: `['len', 'filter', 'gt0']` (3 transitions)
  - `T08_sum_doubled`: `['fold', 'add', 'map', 'mul2']` (4 transitions)
  - `T09_add_two_all`: `['map', 'add2']` (2 transitions)
  - `T10_filter_ge0`: `['filter', 'ge0']` (2 transitions)
  - **Total Training Transitions:** $2 + 1 + 2 + 2 + 2 + 2 + 3 + 4 + 2 + 2 = 22$.
  $$\mathcal{L}(\theta) = -\frac{1}{22} \sum_{t=1}^{22} \ln \left( \frac{\exp(z_{a_t^*})}{\sum_{a' \in \mathcal{A}_{\text{valid}}(\pi_t)} \exp(z_{a'})} \right)$$
- **Optimizer & Training Seed:**
  - Optimizer: AdamW ($\text{lr} = 1.0 \times 10^{-3}$, weight decay $= 1.0 \times 10^{-2}$, cosine annealing schedule with $T_{\max} = 50$ epochs decaying to $\eta_{\min} = 1.0 \times 10^{-4}$).
  - Epochs: 50 epochs (full-batch across the 22 transitions).
  - Training Seed: Strictly pinned at `PRIOR_TRAIN_SEED = 20261026`.

### 5.5. Continuous Metric Kernel Repulsion
Memory uses **max aggregation with bounded normalization**:
$$R(\mathbf{z}_q(P)) = \max_{k \in \mathcal{M}} \left[ \exp\left( \frac{\cos(\mathbf{z}_q(P), \mathbf{z}_k) - 1.0}{\sigma^2} \right) \right] \in [0.0, 1.0], \qquad \sigma = 0.35$$
If $\mathcal{M}$ is empty, $R(\mathbf{z}_q(P)) \equiv 0.0$.
- **Bounded Impact:** Repulsion penalty is strictly bounded by $\beta \cdot 1.0 = 1.0$, preventing penalty runaway.

---

## 6. Frozen 20-Task Family & Task Ranks

All ranks below are taken verbatim from `logs/path_a_feasibility_20261026.txt` and asserted in `tests/test_path_a_feasibility.py`.

### 6.1. Calibration / Training Set (10 Tasks)
| Task Name | Depth | $R_{\text{syn}}$ (Primary) | $R_{\text{dedup}}$ (Sensitivity) | Canonical AST Solution |
| :--- | :---: | :---: | :---: | :--- |
| `T01_sum` | 1 | 17 | 16 | `fold(0, lambda acc, x: acc + x, xs)` |
| `T02_count` | 1 | 2 | 2 | `len(xs)` |
| `T03_double_all` | 1 | 8 | 7 | `map(lambda x: (x * 2), xs)` |
| `T04_add_one` | 1 | 4 | 3 | `map(lambda x: (x + 1), xs)` |
| `T05_filter_positive` | 1 | 11 | 10 | `filter(lambda x: (x > 0), xs)` |
| `T06_filter_evens` | 1 | 14 | 13 | `filter(lambda x: (x % 2 == 0), xs)` |
| `T07_count_positive` | 2 | 28 | 20 | `len(filter(lambda x: (x > 0), xs))` |
| `T08_sum_doubled` | 2 | 235 | 117 | `fold(0, lambda acc, x: acc + x, map(lambda x: (x * 2), xs))` |
| `T09_add_two_all` | 1 | 5 | 4 | `map(lambda x: (x + 2), xs)` |
| `T10_filter_ge0` | 1 | 12 | 11 | `filter(lambda x: (x >= 0), xs)` |

### 6.2. Held-Out Evaluation Set (10 Tasks)
Completely withheld during training; evaluated strictly *zero-shot*:

| Task Name | Depth | $R_{\text{syn}}$ (Primary) | $R_{\text{dedup}}$ (Sensitivity) | Canonical AST Solution |
| :--- | :---: | :---: | :---: | :--- |
| `H01_filter_odds` | 1 | 15 | 14 | `filter(lambda x: (not (x % 2 == 0)), xs)` |
| `H02_double_evens` | 2 | 115 | 74 | `map(lambda x: (x * 2), filter(lambda x: (x % 2 == 0), xs))` |
| `H03_add_two_positives` | 2 | 70 | 39 | `map(lambda x: (x + 2), filter(lambda x: (x > 0), xs))` |
| `H04_count_evens` | 2 | 31 | 23 | `len(filter(lambda x: (x % 2 == 0), xs))` |
| `H05_count_odds` | 2 | 32 | 24 | `len(filter(lambda x: (not (x % 2 == 0)), xs))` |
| `H06_sum_evens` | 2 | 241 | 119 | `fold(0, lambda acc, x: acc + x, filter(lambda x: (x % 2 == 0), xs))` |
| `H07_sum_positives` | 2 | 238 | 116 | `fold(0, lambda acc, x: acc + x, filter(lambda x: (x > 0), xs))` |
| `H08_double_positive_evens`| 3 | **1571** | **395** | `map(lambda x: (x * 2), filter(lambda x: (x > 0), filter(lambda x: (x % 2 == 0), xs)))` |
| `H09_sum_doubled_evens` | 3 | **3293** | **543** | `fold(0, lambda acc, x: acc + x, map(lambda x: (x * 2), filter(lambda x: (x % 2 == 0), xs)))` |
| `H10_sum_add1_evens` | 3 | **3237** | **503** | `fold(0, lambda acc, x: acc + x, map(lambda x: (x + 1), filter(lambda x: (x % 2 == 0), xs)))` |

### 6.3. Headroom under Pinned Canonical Order
- **Primary Syntactic Metric ($R_{\text{syn}}$):**
  - Solved at $K = 500$: 7 of 10 tasks (H01–H07 have $R_{\text{syn}} \le 241$).
  - Unsolved at $K = 500$: **3 of 10 held-out tasks are unsolved by Enumeration within $K = 500$** (H08 at 1571, H10 at 3237, H09 at 3293).
  - The Depth-3 tasks exceed the budget by a margin of 1,071 to 2,793 program evaluations.
- **Secondary Dedup Sensitivity Analysis ($R_{\text{dedup}}$):**
  - Conditionally solves 8 of 10 tasks ($K \le 500$, failing H10 at 503 and H09 at 543) *if and only if* reference-interpreter evaluations are assumed free of compute budget. If reference-interpreter evaluations are charged to budget, dedup execution counts are identical to syntactic evaluations.

---

## 7. Gate 1: World Model / Prior Validation Gate

Before running the search benchmark, the neural prior must be certified on an independent prefix dataset:

- **Definition of a Transition:** Exactly one primitive choice in the canonical prefix traversal (pre-order AST expansion).
- **Exact Gate 1 Sampling Procedure:**
  - Sampler: Python standard library `random.Random` with master seed `GATE1_SAMPLE_SEED = 20261026`.
  - Eligible candidate pools sorted deterministically by `canonical_key()`:
    - Pool D1: All Depth-1 ASTs excluding calibration solutions and held-out solution H01 (exactly 9 available).
    - Pool D2: All Depth-2 ASTs excluding calibration solutions and held-out solutions H02–H07 (exactly 244 available).
    - Pool D3: All Depth-3 ASTs excluding held-out solutions H08–H10 and any program violating the maximal Depth-2 subtree exclusion filter below (exactly 3,456 available).
  - Sample sizes: 5 programs drawn from Pool D1, 20 programs drawn from Pool D2, and 25 programs drawn from Pool D3 (total $N = 50$ independent programs).
- **Maximal Depth-2 Subtree Exclusion Filter:**
  - Definition: Every Depth-3 AST has the structure $C_{\text{outer}}(\text{param}, \text{child})$ where $\text{child}$ is an immediate Depth-2 list sub-expression. The "maximal Depth-2 subtree" is defined as this $\text{child}$ expression.
  - Exclusion Rule: A synthetic Depth-3 program is strictly rejected if its $\text{child}$ sub-expression matches any held-out task's maximal Depth-2 list expression:
    1. `filter(lambda x: (x > 0), filter(lambda x: (x % 2 == 0), xs))` (maximal Depth-2 subtree of H08)
    2. `map(lambda x: (x * 2), filter(lambda x: (x % 2 == 0), xs))` (maximal Depth-2 subtree of H09 and body of H02)
    3. `map(lambda x: (x + 1), filter(lambda x: (x % 2 == 0), xs))` (maximal Depth-2 subtree of H10)
    4. `map(lambda x: (x + 2), filter(lambda x: (x > 0), xs))` (body of H03)
    5. `filter(lambda x: (x % 2 == 0), xs)` / `filter(lambda x: (not (x % 2 == 0)), xs)` / `filter(lambda x: (x > 0), xs)` (subtrees of H04, H05, H06, H07)
  - Complete programs: No sampled program may equal any held-out solution: $\forall P^* \in \mathcal{S}_{\text{heldout}}, P_{\text{synth}} \ne P^*$.
- **Exact Derivation of the 239 Gate 1 Transitions:**
  - Programs 1–5 (5 Depth 1 programs $\times$ 2 steps) = 10 transitions (all at Depth 0 $\to$ 1).
  - Programs 6–25 (20 Depth 2 programs: 19 programs $\times$ 4 steps + 1 program [`len`] $\times$ 3 steps) = 79 transitions (39 at Depth 0 $\to$ 1, 40 at Depth 1 $\to$ 2).
  - Programs 26–50 (25 Depth 3 programs $\times$ 6 steps) = 150 transitions (50 at Depth 0 $\to$ 1, 50 at Depth 1 $\to$ 2, **exactly 50 at Depth 2 $\to$ 3**).
  - **Total Gate 1 Transitions:** $10 + 79 + 150 = 239$ transitions (Depth 0 $\to$ 1: 99, Depth 1 $\to$ 2: 90, Depth 2 $\to$ 3: 50).
- **Clustered Program-Level Statistical Certification Procedure:**
  To eliminate pseudoreplication, testing clusters by program ($N = 50$ independent units):
  - For each synthetic program $j \in \{1, \dots, 50\}$, compute program-level accuracy $Y_j = \frac{1}{|\mathcal{T}_j|} \sum_{(\pi, a^*) \in \mathcal{T}_j} \mathbb{I}(\hat{a}(\pi) = a^*)$ and expected type-valid chance baseline $C_j = \frac{1}{|\mathcal{T}_j|} \sum_{(\pi, a^*) \in \mathcal{T}_j} \frac{1}{|\mathcal{A}_{\text{valid}}(\pi)|}$.
  - Compute paired difference $\Delta_j = Y_j - C_j$ for each of the 50 programs.
  - Observed test statistic: $T_{\text{obs}} = \frac{1}{50} \sum_{j=1}^{50} \Delta_j$.
  - **Monte Carlo Paired Permutation Test ($B = 10,000$):**
    - Fixed seed: `PERM_SEED = 20261026`. Initialize `rng = random.Random(PERM_SEED)`.
    - For $b = 1, \dots, 10000$: draw signs $s_j \in \{-1, +1\}$ uniformly via `rng.choice([-1, 1])`, computing $T^{(b)} = \frac{1}{50} \sum_{j=1}^{50} s_j \cdot \Delta_j$.
    - One-sided empirical p-value: $p = \frac{1 + \sum_{b=1}^{10000} \mathbb{I}(T^{(b)} \ge T_{\text{obs}})}{10001}$.
  - **Decision Rule:** Prior passes Gate 1 if and only if $p < 0.01$ AND aggregate Top-1 accuracy across all 239 transitions $\ge 25.0\%$.

---

## 8. Budget Accounting, Search Cost, Hidden Generalization & Arms

### 8.1. Budget Unit & Search Cost Accounting
- **Primary Budget Unit ($E \in \mathbb{N}$):** Running one complete candidate AST on all 5 visible test cases in the isolated subprocess sandbox. **This primary metric applies identically to all three arms: every candidate program executed consumes exactly 1 budget unit.** Behavioral deduplication is strictly a secondary sensitivity analysis ($R_{\text{dedup}}$) and does not modify the primary budget accounting for any arm.
- **Budget Cap ($K$):** Exactly **$K = 500$ program executions** per task. Unsolved tasks are penalized at $E = 500$.
- **Secondary Search Cost Metrics:**
  1. **Total Prefix Expansions ($X \in \mathbb{N}$):** Number of priority queue pop/expansion operations.
  2. **Wall-Clock Search Time ($T \in \mathbb{R}$ seconds):** Total elapsed execution and inference time per task.

### 8.2. Pinned Hidden Generalization Suite (Overfitting Guard)
Every candidate program flagged as "solved" on visible cases is evaluated on a **Hidden Suite $\mathcal{E}_{\text{hidden}}$ of 20 unobserved test inputs**:
- **Generator Specification:** Python `random.Random` with master seed `MASTER_HIDDEN_SEED = 20261026`.
- **Explicit Held-Out Task Sequence:** The task index $k \in \{0, \dots, 9\}$ is defined by the fixed benchmark list:
  `0: H01_filter_odds`, `1: H02_double_evens`, `2: H03_add_two_positives`, `3: H04_count_evens`, `4: H05_count_odds`, `5: H06_sum_evens`, `6: H07_sum_positives`, `7: H08_double_positive_evens`, `8: H09_sum_doubled_evens`, `9: H10_sum_add1_evens`.
- **Per-Task Stream Derivation:** For task index $k \in \{0, \dots, 9\}$, `task_seed = MASTER_HIDDEN_SEED + 1000 * (k + 1)`. Initialize `rng = random.Random(task_seed)`.
- **Exact Call Order per Task:**
  - 9 pinned edge cases: `[]`, `[5]`, `[-4]`, `[0]`, `[0, 0, 0]`, `[2, 4, -6, 8]`, `[1, 3, -5, 7]`, `[-3, 4, -5, 6, -7]`, `[2, 2, -2, -2]`.
  - 11 randomized vectors generated sequentially:
    ```python
    for _ in range(11):
        length = rng.randint(1, 8)
        inp = [rng.randint(-10, 10) for _ in range(length)]
        hidden_inputs.append(inp)
    ```
- **Verification Rule:** Candidate must achieve 20/20 (100%) accuracy on $\mathcal{E}_{\text{hidden}}$ against independent task reference implementations.

### 8.3. Experimental Arms
1. **Arm 1 (Uniform Canonical Enumeration):** Deterministic BFS following frozen total-order comparator (Section 3).
2. **Arm 2 (Learned Prior Search without Memory):** Learned prior only ($\beta = 0.0$).
3. **Arm 3 (Full Earthos Substrate):** Learned prior + failure attractor memory ($\beta = 1.0$).

---

## 9. Success Criteria & Pre-Registered Decision Rules

### 9.1. Descriptive Pilot Success Criteria
Earthos Path A is considered promising if:
1. **Task Solve Rate:** Earthos solves strictly more tasks than Uniform Enumeration ($N_{\text{solved}}^{\text{Earthos}} > N_{\text{solved}}^{\text{Enum}}$) within budget $K = 500$ under the primary Syntactic Metric.
2. **Sample Efficiency:** Mean executions across the 10 tasks is lower for Earthos: $\bar{E}_{\text{Earthos}} < \bar{E}_{\text{Enum}}$.
3. **Memory Efficacy:** Full Earthos with Memory requires fewer mean executions than Memory Ablation: $\bar{E}_{\text{Memory}} < \bar{E}_{\text{Ablation}}$.

### 9.2. Status of Ties (Inconclusive Non-Success)
A tie on solved task count ($N_{\text{solved}}^{\text{Earthos}} = N_{\text{solved}}^{\text{Enum}}$) is designated as an **Inconclusive Non-Success**, not an outright falsification, provided Earthos demonstrates a notable sample efficiency advantage:
$$\bar{E}_{\text{Earthos}} < 0.80 \cdot \bar{E}_{\text{Enum}}$$

### 9.3. Mixed Result (No Superiority Claim)
If Earthos solves strictly more tasks than Uniform Enumeration ($N_{\text{solved}}^{\text{Earthos}} > N_{\text{solved}}^{\text{Enum}}$) within budget $K = 500$, but has equal or higher mean execution count ($\bar{E}_{\text{Earthos}} \ge \bar{E}_{\text{Enum}}$), this outcome is designated as a **Mixed Result, reported descriptively with no superiority claim**.

### 9.4. Pre-Registered Failure Conditions (Reconciled & Exhaustive)
Earthos Path A will be declared **FAILED** if any of the following mutually exclusive falsification conditions occur:
1. **Enumeration Dominance:** Uniform Enumeration solves strictly more tasks than Earthos within budget $K = 500$ ($N_{\text{solved}}^{\text{Enum}} > N_{\text{solved}}^{\text{Earthos}}$).
2. **Tie without Sample Efficiency Advantage:** Earthos and Enumeration solve the exact same number of tasks ($N_{\text{solved}}^{\text{Earthos}} = N_{\text{solved}}^{\text{Enum}}$), and Earthos fails to achieve the required $\ge 20\%$ efficiency advantage ($\bar{E}_{\text{Earthos}} \ge 0.80 \cdot \bar{E}_{\text{Enum}}$).
3. **Memory Inefficacy:** Full Earthos with Memory does not outperform Memory Ablation ($\bar{E}_{\text{Memory}} \ge \bar{E}_{\text{Ablation}}$).

### 9.5. Pre-Registration of Small Memory Effect (A Priori Fixed Boundaries)
Because failure traces in Arm 3 only repel prefixes that structurally resonate with failed candidates, and many random failed candidates do not share subtrees with the true solution, the marginal gain of Arm 3 over Arm 2 ($\bar{E}_{\text{Memory}}$ vs $\bar{E}_{\text{Ablation}}$) is expected to be modest in an $N=10$ pilot. The following boundaries are fixed a priori:
- **Modest Substrate Effect:** Defined as $0.85 \le \bar{E}_{\text{Memory}} / \bar{E}_{\text{Ablation}} < 1.00$ (0% to 15% execution reduction).
- **Strong Substrate Effect:** Defined as $\bar{E}_{\text{Memory}} / \bar{E}_{\text{Ablation}} < 0.85$ (>15% execution reduction).
- **Memory Inefficacy (Failure):** $\bar{E}_{\text{Memory}} / \bar{E}_{\text{Ablation}} \ge 1.00$.

---

## 10. Non-Circular Commit Pinning Protocol & Test Suites

- **Non-Circular Commit Pinning Protocol:**
  1. The benchmark assets (DSL, comparator, 20 tasks, feasibility script, feasibility log, and feasibility unit tests) are committed in the Proposed Freeze Commit ($C_{\text{freeze}}$).
  2. The exact SHA of $C_{\text{freeze}}$ and SHA256 hashes of the assets are recorded in `docs/prereg/path_a_preregistration.md`.
- **Canonical Regression Suite (58/58 Passing):**
  - `tests/test_substrate_canonical.py`: 7 tests
  - `tests/test_substrate_checkpoint.py`: 7 tests
  - `tests/test_substrate_chain.py`: 5 tests
  - `tests/test_substrate_newline.py`: 3 tests
  - `tests/test_substrate_rng.py`: 3 tests
  - `tests/test_path_a_feasibility.py`: 4 tests
  - `earthoslab/test_neurosymbolic_synthesis.py`: 6 tests
  - `earthoslab/test_smoke.py`: 2 tests
  - `tests/test_substrate_regression.py`: 21 tests
  - **Total:** Exactly **58 passed** (deterministic across all runs; execution time varies between ~17s and ~43s depending on machine subprocess load).

---
*Document proposed for supervisor review. Feasibility check verified via `scripts/path_a_feasibility_check.py` and `tests/test_path_a_feasibility.py`.*

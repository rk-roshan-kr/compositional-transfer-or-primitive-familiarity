# Publication Checklist & Operational Guide: Paper 11

**Paper Title:** Compositional Transfer or Primitive Familiarity? Dissecting Prior Guidance in Neurosymbolic Program Synthesis  
**Target Venues:** NeurIPS / ICLR (Primary: Main Track / Benchmark Track)  
**Submission Readiness:** **95%** (Code, logs, controls, and manuscript draft complete)  

---

## 1. Introductory Operational Guide

### 1.1. Purpose of this Checklist
This checklist is organized into **thematic blocks** that directly mirror the progressive lifecycle of a scientific study: from conceptualization and baseline establishment to causal isolation, statistical verification, manuscript drafting, and reproducibility artifact bundling.

### 1.2. How to Use this Checklist Across Research Phases
Researchers and autonomous build agents should apply this guide sequentially:

```
[Phase A: Scope & Invariants] ──> [Phase B: Experimental Isolation] ──> [Phase C: Verification & Generalization] ──> [Phase D: Manuscript & Ledger]
```

1. **Phase A (Scope & Invariants):**
   * Fix the formal DSL grammar, primitive counts, and depth definitions before any search runs.
   * Lock down deterministic total-order comparators to ensure baseline enumeration is 100% reproducible.
   * Preregister failure conditions and budget limits ($K = 500$) to prevent post-hoc goalpost shifting.
2. **Phase B (Experimental Isolation):**
   * Never compare a guided search arm directly to a different algorithmic baseline without an ablation control.
   * Implement the **Uniform Prior control** first to quantify how much of the search speedup is driven by the priority queue and length penalty alone.
   * Implement the **Unigram Prior control** to decouple marginal primitive popularity from structural grammar.
   * Evaluate **Unseen Primitives** to test whether compositional transfer holds outside the training vocabulary.
3. **Phase C (Verification & Generalization):**
   * Run candidate solutions through an isolated, multi-input hidden evaluation suite.
   * Verify zero semantic false positives (20/20 test pass required).
   * Confirm type safety across all score/depth ties in the priority queue.
4. **Phase D (Manuscript & Ledger Freeze):**
   * Anchor all experimental numbers to verbatim saved log files.
   * Update the internal scope ledger (`ledger.md`) with artifact paths and git commit hashes.
   * Audit all claims to ensure findings are reported descriptively without statistical inflation.

---

## 2. Thematic Block Publication Checklist

### Block 1: Problem Formulation & Core Research Questions
- [x] **Primary Research Questions Formulated:** $RQ_1$ (Search Attribution: queue vs. unigram vs. context), $RQ_2$ (Out-of-Vocabulary Transfer: novel primitives generalization vs. negative transfer).
- [x] **Formal DSL Syntax & Semantics Defined:** 21 typed primitives (4 combinators, 7 scalar ops, 7 predicates, 3 reducers).
- [x] **Program Space Cardinality Proven:** Total space bounded at exactly 3,799 programs for Depth $\le 3$.
- [x] **Total-Order Comparator Specified:** Injective tuple comparator eliminating tie ambiguity in enumeration.

### Block 2: Baselines & Experimental Isolation Arms
- [x] **Arm 1 (Canonical Enumeration):** Deterministic BFS baseline confirming 7/10 solved, mean 224.2 executions, Depth-3 unreachable within $K = 500$.
- [x] **Arm 2a (Uniform Prior Control):** Same Best-First search engine ($\lambda = 0.05$), equal action probabilities ($1 / |\mathcal{A}|$). Solves 9/10, fails H08.
- [x] **Arm 2b (Unigram Marginal Prior Control):** Frequency-only prior ignoring context. Matches H08 at step 244, solves 10/10, mean 63.8 executions.
- [x] **Arm 2c (Bigram Prior, Free Close):** Context-conditioned prior with free list closure. Solves 10/10, mean 57.1 executions.
- [x] **Arm 2d (Bigram Prior, Priced Close):** List closure penalized by $-\lambda$. Proves identical 57.1 mean, ruling out close-action bias.

### Block 3: Compositional Transfer & Out-of-Distribution Testing
- [x] **Unseen Primitives Benchmark (U01–U04):** Tasks constructed exclusively from primitives with zero training count (`mul3`, `sub1`, `mod3_0`, `odd`).
- [x] **Negative Transfer Quantified:** Learned prior verified slower than uniform search (mean 217.2 vs. 187.2 executions).
- [x] **Decision-Level Backoff Accounting:** Renamed to `D1_context`, `D2_slot`, `D3_uniform` and counted once per expansion decision.
- [x] **Depth-3 Diagnostic Reading:** Reported descriptively with intra-program dependence disclaimer.

### Block 4: Generalization & Semantic Verification
- [x] **Pinned Hidden Suite Generator:** Master seed `20261026` + task seed derivation according to Section 8.2.
- [x] **Edge Case Invariance:** 9 boundary cases (`[]`, `[0]`, `[-4]`, `[0, 0, 0]`, etc.) verified.
- [x] **Randomized Vector Stress Testing:** 11 randomized vectors of varying lengths (1–8) verified.
- [x] **100% Pass Rate:** All discovered solutions across all arms achieved 20/20 against pure-Python references.

### Block 5: Software & Heap Robustness
- [x] **Heap Key Homogeneity:** Unified tuple `(depth, length, is_partial, prims_tuple, item_id)` preventing `TypeError` on ties.
- [x] **Monotonic Tie-Breaker:** Strict FIFO integer sequence counter implemented.
- [x] **Unit Test Suite:** `tests/test_path_a_feasibility.py` 4/4 passing in 0.13s.

### Block 6: Manuscript & Artifact Ledger
- [x] **Full Manuscript Written:** `docs/papers/compositional_transfer_or_primitive_familiarity.md`.
- [x] **Internal Scope Ledger Established:** `papers/11_compositional_transfer_or_primitive_familiarity/ledger.md`.
- [x] **Verbatim Terminal Logs Saved:** `logs/prior_controls_20261026.txt`.
- [x] **Task History Audited:** `TASK-43.12` recorded in `logs/task_history.md`.
- [x] **Submission Readiness Score:** **95% / 100%**.

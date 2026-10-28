# Path A Proposed Pre-Registration Record
**File:** `docs/prereg/path_a_preregistration.md`  
**Date:** 2026-10-26  
**Status:** PROPOSED PRE-REGISTRATION RECORD (Under Supervisor Review)  
**Proposed Freeze Commit Hash ($C_{\text{freeze}}$):** `e32fd5768e9d6dc17e8ef4b75e8b8c09fe7132e4` (short: `e32fd57`)  

---

## 1. Immutable Asset Hashes (Computed Directly from Commit `e32fd57` Tree)

All hashes below are computed directly from git tree objects using `git show e32fd57:<path> | sha256sum`, guaranteeing reproducibility independent of working-tree line-ending conversions:

| File Path | Commit Tree | SHA256 Hash | Purpose |
| :--- | :---: | :--- | :--- |
| `docs/design/path_a_neurosymbolic_synthesis.md` | `e32fd57` | `190acfb581ad8b64ab421c1f9f0134a4589c78c070b0f7564dbbe9296a80f484` | Master design specification & decision rules |
| `scripts/path_a_feasibility_check.py` | `e32fd57` | `48a618baafed2f1616f6ba419245c735b269463de2b88750a2ee8a8caaa6f18a` | Canonical total-order comparator & search space audit |
| `logs/path_a_feasibility_20261026.txt` | `e32fd57` | `80ec484b15c061e4b1d275d324e6de0b034e9e188d2711194f4b8e093e1660d3` | Frozen syntactic & dedup rank output log |
| `tests/test_path_a_feasibility.py` | `e32fd57` | `45d7ba3b9ce0019a2e0f46d672319d3aa510c9144d5b0ec28c387091e700662f` | Mathematical unit tests (counts, injectivity, ranks, references) |

---

## 2. Technical Justification for Repository Cleanliness & Historical Context

1. **Repository History & Files Outside Protocol:**
   - Commit `915b31e` included pre-existing untracked exploratory research artifacts (`EARTHOS_FILE_ANALYSIS.md`, chess comparison PDF/markdown, startup blueprints, reference architecture notes, and earlier drafts). These files are historical exploratory records and are **explicitly designated as outside the Path A benchmark protocol**.
2. **Unstaging `reffernce/laya/` from Index and Updating `.gitignore`:**
   - **Reason:** `reffernce/laya/` is an external research reference clone that contained its own `.git` directory. Running `git add .` staged it as an unregistered gitlink, triggering corrupted subproject warnings. To prevent broken tree links, `git rm --cached -f reffernce/laya` unindexed the directory, and `reffernce/laya/` was added to `.gitignore`.
3. **Restoring `reffernce/claudecode/tsconfig.json` & Submodule Config:**
   - **Reason:** A local read had dirtied `tsconfig.json` in `reffernce/claudecode`. `git -C reffernce/claudecode checkout tsconfig.json` restored the submodule to commit `126c311`.
   - `git config diff.ignoreSubmodules dirty` was configured locally to prevent untracked scratch files inside submodules from dirtying repository status.
4. **Core Integrity:** Zero core substrate files, test suites, or benchmark assets were touched. `git status --porcelain` is clean.

---

## 3. Pinned Benchmark Specifications

1. **Action Vocabulary (21 Primitives):**
   - 4 Combinators: `map`, `filter`, `fold`, `len`
   - 7 Scalar Operations: `id`, `add1`, `add2`, `add3`, `sub1`, `mul2`, `mul3`
   - 7 Predicates: `eq0`, `gt0`, `ge0`, `lt0`, `even`, `odd`, `mod3_0`
   - 3 Reducers: `add`, `sub`, `mul`
   - Terminal: `xs` (base input list variable)
   - *(Note: The count 18 is the number of distinct complete Depth-1 programs over `xs`, not the primitive count).*
2. **Primary Budget Metric Invariance Across All Arms:**
   - The primary metric counts every candidate program execution in the isolated sandbox across all three arms ($E_i \in \mathbb{N}$). Behavioral deduplication ($R_{\text{dedup}}$) is strictly a secondary sensitivity analysis and does not alter primary budget accounting.
3. **Baseline Headroom:**
   - Exactly 3 of 10 held-out tasks are unsolved by Uniform Enumeration within $K = 500$ under the pinned canonical order (H08 at 1571, H10 at 3237, H09 at 3293).
4. **Prior Architecture & Calibration Training:**
   - Spec Features: 16-dim $\mathbf{e}_{\text{spec}}(\mathcal{E})$.
   - AST Features: 16-dim $\mathbf{e}_{\text{AST}}(\pi)$.
   - Architecture: 2-layer MLP ($32 \to 64 \to 64 \to 21$) with LayerNorm and GELU.
   - Output Head: $\text{Linear}(64 \to 21)$ covering all 21 action primitives.
   - Calibration Training Transitions: Exactly 22 transitions across the 10 calibration tasks (T01: 2, T02: 1, T03: 2, T04: 2, T05: 2, T06: 2, T07: 3, T08: 4, T09: 2, T10: 2).
   - Optimizer: AdamW ($\text{lr} = 1.0 \times 10^{-3}$, weight decay $= 1.0 \times 10^{-2}$, cosine annealing with $T_{\max} = 50$ epochs decaying to $\eta_{\min} = 1.0 \times 10^{-4}$). Pinned seed: `PRIOR_TRAIN_SEED = 20261026`.
5. **Gate 1 Validation Protocol:**
   - Sampler: `random.Random(20261026)` drawing 5 Depth 1, 20 Depth 2, and 25 Depth 3 programs from deterministic pools sorted by `canonical_key()`.
   - Gate 1 Transitions: Exactly 239 transitions (10 at Depth 1, 79 at Depth 2, 150 at Depth 3; 50 at Depth 2 $\to$ 3).
   - Exclusion Filter: Complete held-out solutions are excluded ($\forall P^* \in \mathcal{S}_{\text{heldout}}, P_{\text{synth}} \ne P^*$). Furthermore, any Depth-3 program containing a maximal Depth-2 list sub-expression of a held-out task (e.g., `filter(gt0, filter(even, xs))` from H08) is strictly rejected.
   - Statistical Certification: One-sided Monte Carlo Paired Permutation Test over 50 program clusters ($B = 10,000$, `PERM_SEED = 20261026`), requiring $p < 0.01$ and aggregate Top-1 accuracy $\ge 25.0\%$.
6. **Pinned Hidden Generalization Suite:**
   - Explicit Task Sequence: `[H01, H02, H03, H04, H05, H06, H07, H08, H09, H10]`.
   - Per-Task Seed: `task_seed = 20261026 + 1000 * (k + 1)` for $k \in \{0, \dots, 9\}$.
   - Call Order: 9 fixed edge cases, followed by 11 sequential random draws with `length = rng.randint(1, 8)` and elements `rng.randint(-10, 10)`.
   - 20/20 (100%) accuracy required against independent reference functions.
7. **Decision Rules & Outcome Classification:**
   - Success: $N_{\text{solved}}^{\text{Earthos}} > N_{\text{solved}}^{\text{Enum}}$ AND $\bar{E}_{\text{Earthos}} < \bar{E}_{\text{Enum}}$ AND $\bar{E}_{\text{Memory}} < \bar{E}_{\text{Ablation}}$.
   - Status of Ties (Inconclusive Non-Success): $N_{\text{solved}}^{\text{Earthos}} = N_{\text{solved}}^{\text{Enum}}$ and $\bar{E}_{\text{Earthos}} < 0.80 \cdot \bar{E}_{\text{Enum}}$.
   - Mixed Result (No Superiority Claim): $N_{\text{solved}}^{\text{Earthos}} > N_{\text{solved}}^{\text{Enum}}$ but $\bar{E}_{\text{Earthos}} \ge \bar{E}_{\text{Enum}}$.
   - Failure: Enumeration dominance ($N_{\text{solved}}^{\text{Enum}} > N_{\text{solved}}^{\text{Earthos}}$), Tie without 20% advantage ($N_{\text{solved}}^{\text{Earthos}} = N_{\text{solved}}^{\text{Enum}}$ and $\bar{E}_{\text{Earthos}} \ge 0.80 \cdot \bar{E}_{\text{Enum}}$), or Memory inefficacy ($\bar{E}_{\text{Memory}} \ge \bar{E}_{\text{Ablation}}$).
   - Modest Memory Effect Boundary: $0.85 \le \bar{E}_{\text{Memory}} / \bar{E}_{\text{Ablation}} < 1.00$ fixed a priori.

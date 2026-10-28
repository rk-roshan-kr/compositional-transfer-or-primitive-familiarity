# Paper 11 Internal Scope Ledger: Dissecting Prior Guidance in Neurosymbolic Program Synthesis

**Ledger ID:** `LEDGER-PAPER-11-20261026`  
**Paper Directory:** `papers/11_compositional_transfer_or_primitive_familiarity/`  
**Manuscript File:** `docs/papers/compositional_transfer_or_primitive_familiarity.md`  
**Status:** COMPLETE / AUDITED / PREREGISTERED  
**Date Established:** 2026-10-26  

---

## 1. Paper Overview & Scope Boundaries

* **Title:** Compositional Transfer or Primitive Familiarity? A Causal Decomposition of Prior-Guided Program Synthesis
* **Core Research Bet:** Subjecting neurosymbolic prior guidance to strict causal isolation controls proves that search acceleration is driven by Best-First priority queue dynamics and unigram primitive familiarity, rather than abstract compositional transfer.
* **Scope Inclusions:**
  * 21-primitive typed list processing DSL ($D \le 3$, 3,799 programs).
  * 10 calibration tasks (`T01`–`T10`, 22 transitions total).
  * 10 held-out evaluation tasks (`H01`–`H10`).
  * 4 unseen primitive tasks (`U01`–`U04`).
  * 4 search arms: Canonical Enumeration, Uniform Best-First, Unigram Best-First, Bigram Best-First (Free Close vs. Priced Close).
  * 20-input pinned hidden generalization suite ($9 \text{ fixed edge cases} + 11 \text{ sequential random vectors}$).
* **Scope Exclusions (Strict Non-Claims):**
  * Neural MLP prior training (omitted; count-based prior serves as analytical gold standard).
  * Memory kernel repulsion ($\beta = 0.0$ throughout; memory arm reserved for future work).
  * Program depths $D > 3$ (strictly out of scope).

---

## 2. File & Artifact Manifest

### 2.1. Core Executable Scripts
| File Path | Role | Description |
| :--- | :--- | :--- |
| `scripts/path_a_feasibility_check.py` | DSL & Space Definition | Defines AST nodes, 21 primitives, evaluation, canonical total-order comparator, and task fixtures. |
| `scripts/enum_pilot.py` | Baseline Harness | Executes canonical uniform enumeration pilot across H01–H10. |
| `scripts/prior_pilot.py` | Preliminary Guided Pilot | Initial Arm 2 pilot implementing count-based bigram prior. |
| `scripts/prior_controls.py` | **Primary Experimental Harness** | Full multi-arm isolation script (Uniform, Unigram, Bigram, Priced Close, Unseen U01–U04, Hidden Suite). |

### 2.2. Unit & Integration Test Suites
| File Path | Tests Passing | Description |
| :--- | :---: | :--- |
| `tests/test_path_a_feasibility.py` | 4 / 4 | Asserts DSL depth/size/rendering, 3,799 program count, total-order injectivity, and canonical ranks. |
| `tests/audit_determinism.py` | Verified | Checks determinism and environment invariants. |

### 2.3. Empirical Logs & Verbatim Data
| File Path | Bytes | Description |
| :--- | :---: | :--- |
| `logs/path_a_feasibility_20261026.txt` | 3,676 | Verbatim log of syntactic and behavioral deduplication ranks for T01–T10 and H01–H10. |
| `logs/enum_pilot_20261026.txt` | 1,993 | Baseline log confirming 7/10 solved, mean 224.2 executions. |
| `logs/prior_pilot_20261026.txt` | 3,968 | Preliminary Arm 2 pilot log showing 10/10 solved, mean 56.2 executions. |
| `logs/prior_controls_20261026.txt` | 5,441 | **Definitive multi-arm isolation log** with all side-by-side tables and 20/20 hidden suite results. |

### 2.4. Specification & Preregistration Documents
| File Path | Status | Role |
| :--- | :---: | :--- |
| `docs/design/path_a_neurosymbolic_synthesis.md` | Frozen | Formal design specification (Sections 1–10). |
| `docs/prereg/path_a_preregistration.md` | Preregistered | Pre-registered protocol, Gate 1 certification rules, and failure conditions. |
| `docs/reports/path_a_prior_controls_report.md` | Complete | Internal investigation report detailing the causal attribution analysis. |
| `docs/papers/compositional_transfer_or_primitive_familiarity.md` | Complete | **Full scientific manuscript (Markdown).** |
| `papers/11_compositional_transfer_or_primitive_familiarity/paper.tex` | Complete | **Full conference manuscript (LaTeX source).** |
| `papers/11_compositional_transfer_or_primitive_familiarity/paper.pdf` | Compiled | **Compiled 6-page conference paper PDF (MiKTeX pdflatex).** |
| `docs/papers/compositional_transfer_or_primitive_familiarity.pdf` | Compiled | **Compiled conference paper PDF (mirror).** |


---

## 3. Pinned Experimental Seeds & Constants

| Constant | Value | Purpose |
| :--- | :--- | :--- |
| `PRIOR_TRAIN_SEED` | `20261026` | Master seed for calibration data partitioning. |
| `GATE1_SAMPLE_SEED` | `20261026` | Master seed for Gate 1 synthetic program sampler (5 D1, 20 D2, 25 D3). |
| `PERM_SEED` | `20261026` | Permutation test seed for Gate 1 Monte Carlo testing ($B = 10,000$). |
| `MASTER_HIDDEN_SEED` | `20261026` | Master seed for 20-input hidden generalization suite. |
| `lambda_val` ($\lambda$) | `0.05` | Parsimony penalty per combinator step. |
| `alpha` ($\alpha$) | `1.0` | Laplace smoothing parameter across all prior models. |
| `budget_k` ($K$) | `500` | Max complete program executions permitted per task. |
| `max_expansions` ($X_{\max}$) | `100000` | Safety cap on priority queue expansions. |

---

## 4. Audit Trail & Git Anchors

* **Base Freeze Commit:** `41346f7`
* **Task History Entries:**
  * `TASK-43.08`: Design document formalization.
  * `TASK-43.09`: Total-order comparator and syntactic audit.
  * `TASK-43.10`: Canonical enumeration baseline pilot execution.
  * `TASK-43.11`: Arm 2 preliminary prior pilot execution.
  * `TASK-43.12`: Multi-arm isolation controls execution and formal causal documentation.

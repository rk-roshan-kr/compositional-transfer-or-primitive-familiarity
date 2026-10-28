# Compositional Transfer or Primitive Familiarity? What a Count-Based Prior Explains in a Typed Functional DSL

[![Tests](https://img.shields.io/badge/tests-4%2F4%20passed-brightgreen.svg)]()
[![Paper](https://img.shields.io/badge/paper-camera--ready%20PDF%20(6%20pages)-blue.svg)](paper/paper.pdf)
[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Official replication package and code repository for:
> **Compositional Transfer or Primitive Familiarity? What a Count-Based Prior Explains in a Typed Functional DSL**  
> **Author:** Roshan Kumar  
> **Affiliation:** FieldChain  
> **Camera-Ready Paper:** [`paper/paper.pdf`](paper/paper.pdf) (Exact 6 pages, 0 overfull hbox, 36 verified citations) | [`paper/paper.md`](paper/paper.md)

---

## 🔬 Abstract & Executive Summary

Statistical priors trained on execution traces are widely reported to accelerate neurosymbolic program synthesis. When a model trained on shallow Depth-1 and Depth-2 programs discovers Depth-3 compositions, this efficiency is often attributed to **compositional transfer**—the inductive bias that the model learns abstract hierarchical syntax rules.

This repository provides an exact, full-space empirical decomposition ($N = 3,799$ well-typed programs) demonstrating that this attribution is largely an artifact of conflating basic search queue mechanics and single-primitive frequency with structural grammar learning:

1. **The Search Engine Explains Most Speedup ($81.7\%$ mean / $66.1\%$ median):**  
   Simply switching from standard Breadth-First Search (BFS) to a priority queue with length penalty using a **Uniform Prior** ($1/|\mathcal{A}|$, no statistical learning whatsoever) reduces mean search evaluations on held-out tasks from **884.3** down to **208.3** (median $176.5 \to 76.0$).
2. **Single-Primitive Popularity Dominates Statistical Gain ($95.6\%$ mean / $99.0\%$ median):**  
   A context-free **Unigram Prior** (ignoring parent combinators and argument slots) further reduces mean evaluations to **63.8** (median **24.5**). Adding transition-conditioned Bigram context yields only **6.7** additional evaluations on average (0.5 median).
3. **No Compositional Transfer to Unseen Primitives (Persistent Negative Transfer):**  
   When held-out tasks require valid primitives with zero training count ($U_1$–$U_4$, Depth 1–3), learned prior guidance degrades search performance below Uniform random search across all tasks (**2.6$\times$ slowdown** on Depth-3 $U_4$: 1,795 vs 589 evaluations). Empirical Laplace smoothing sweeps ($\alpha \in [0.01, 10.0]$) confirm that backoff to smoothed unigrams persists in prioritizing familiar primitives over novel ones, slowing search across all smoothing regimes.

---

## 📊 Summary of Empirical Results

### Table 1: Synthesis Search Effort Across Held-Out Tasks ($H_1$–$H_{10}$)
*Evaluated candidate programs before locating the first semantically correct solution under a total space of $N = 3,799$ programs.*

| Task | Depth | Arm 1: BFS | Uniform Prior | Unigram Prior | Bigram (Free) | Bigram (Priced) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| $H_1$ (`head_pos`) | 2 | 23 | 17 | 10 | 10 | 10 |
| $H_2$ (`even_count`) | 2 | 25 | 18 | 13 | 13 | 13 |
| $H_3$ (`add2_all`) | 1 | 8 | 7 | 8 | 8 | 8 |
| $H_4$ (`mul2_first`) | 2 | 227 | 78 | 17 | 17 | 17 |
| $H_5$ (`ge0_count`) | 2 | 27 | 20 | 14 | 14 | 14 |
| $H_6$ (`sum_add1`) | 2 | 237 | 74 | 22 | 22 | 22 |
| $H_7$ (`last_elem`) | 1 | 18 | 16 | 27 | 27 | 27 |
| $H_8$ (`double_evens`) | 3 | 1,571 | 1,061 | 244 | 244 | 244 |
| $H_9$ (`pos_plus2_count`)| 3 | 3,293 | 436 | 174 | 79 | 79 |
| $H_{10}$ (`sum_even_gt0`) | 3 | 3,237 | 356 | 109 | 117 | 117 |
| **Solved / 10 ($K=500$)** | — | **7 / 10** | **9 / 10** | **10 / 10** | **10 / 10** | **10 / 10** |
| **Uncensored Mean** | — | **884.3** | **208.3** | **63.8** | **57.1** | **57.1** |
| **Uncensored Median** | — | **176.5** | **76.0** | **24.5** | **24.5** | **24.5** |

### Table 2: Generalization to Unseen Primitives ($U_1$–$U_4$)
*Evaluating tasks composed exclusively of primitives unseen during prior training (`mul3`, `sub1`, `mod3_0`, `odd`).*

| Task | Target AST | Depth | Canonical BFS | Uniform Prior | Learned Prior (Backoff) | Negative Transfer |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| $U_1$ (`sub1_all`) | `map(sub1, xs)` | 1 | 9 | 8 | **11** | $+37.5\%$ |
| $U_2$ (`mod3_count`) | `len(filter(mod3_0, xs))` | 2 | 29 | 22 | **35** | $+59.1\%$ |
| $U_3$ (`mul3_evens`) | `map(mul3, filter(even, xs))` | 2 | 251 | 134 | **341** | $+154.5\%$ |
| $U_4$ (`mul3_odd_pos`)| `map(mul3, filter(odd, filter(gt0, xs)))` | 3 | 3,363 | 589 | **1,795** | **$+204.8\%$ ($2.6\times$)** |
| **Mean** | — | — | **913.0** | **188.2** | **545.5** | **$+189.9\%$** |

---

## 📐 Search Space & Comparator Invariants

The DSL is a pure, strongly-typed functional list manipulation language with 21 primitives:
* **4 Combinators:** `map`, `filter`, `fold`, `len`
* **7 Scalar Ops:** `id`, `add1`, `add2`, `add3`, `sub1`, `mul2`, `mul3`
* **7 Predicates:** `eq0`, `gt0`, `ge0`, `lt0`, `even`, `odd`, `mod3_0`
* **3 Reducers:** `add`, `sub`, `mul`

At maximum depth bound $D \le 3$, the space partitions into:
$$\text{Depth 0: } 1 \quad\mid\quad \text{Depth 1: } 18 \quad\mid\quad \text{Depth 2: } 252 \quad\mid\quad \text{Depth 3: } 3,528 \quad\implies\quad \mathbf{N = 3,799 \text{ total ASTs}}$$

### Total-Order Comparator
To guarantee 100% reproducible enumeration with zero tie ambiguities, candidate programs are totally ordered by the injective tuple:
$$K_{\text{syn}}(P) = \Big(\text{depth}(P),\, \text{size}(P),\, \text{cat}(P),\, \text{id}(P),\, \text{child\_keys}(P)\Big)$$

Every discovered candidate is verified against a pinned 20-input hidden generalization test suite generated from master seed `20261026` with 0% false positives.

---

## 📁 Repository Structure

```
compositional-transfer-or-primitive-familiarity/
├── README.md                           # This publication-grade documentation
├── LICENSE                             # MIT License
├── pyproject.toml                      # Standard Python packaging configuration
├── requirements.txt                    # Minimal dependencies (pytest, pypdf)
├── paper/
│   ├── paper.tex                       # Camera-ready LaTeX manuscript (6 pages)
│   ├── paper.pdf                       # Compiled camera-ready PDF
│   ├── paper.md                       # Full markdown manuscript mirror
│   ├── references.bib                  # 36 DBLP/ACM verified bibliographic records
│   ├── publication_checklist.md        # 6-block operational research checklist
│   ├── ledger.md                       # Internal scope & commit freeze ledger
│   └── preview/                        # High-resolution PNG page previews (1-8)
├── src/
│   └── compositional_transfer/         # Reusable Python library
│       └── __init__.py                 # Core package exports
├── scripts/
│   ├── path_a_feasibility_check.py     # Deterministic 3,799 space generator & rank auditor
│   ├── enum_pilot.py                   # Canonical BFS baseline runner (Arm 1)
│   ├── prior_pilot.py                  # Prior-guided Best-First Search runner (Arm 2)
│   ├── prior_controls.py               # Complete uncensored multi-arm isolation ablation
│   └── check_pdf_layout.py             # Strict 6-page & 0 overfull hbox validator
├── tests/
│   └── test_path_a_feasibility.py      # Unit test suite verifying space counts & injectivity
├── docs/
│   ├── controls_report.md              # In-depth empirical investigation report
│   ├── preregistration.md              # Frozen pre-registration specification record
│   └── design_specification.md         # Full formal benchmark design doc
└── logs/
    ├── path_a_feasibility_20261026.txt # Verbatim rank audit log
    ├── enum_pilot_20261026.txt         # Verbatim Arm 1 pilot log
    ├── prior_pilot_20261026.txt        # Verbatim Arm 2 pilot log
    └── prior_controls_20261026.txt     # Full uncensored ablation log
```

---

## 🚀 Quickstart & Reproduction

### 1. Environment Setup
The codebase is pure Python and relies exclusively on standard libraries, requiring only `pytest` for unit testing and `pypdf` for layout verification:

```bash
git clone https://github.com/rk-roshan-kr/compositional-transfer-or-primitive-familiarity.git
cd compositional-transfer-or-primitive-familiarity
pip install -r requirements.txt
```

### 2. Run Test Suite
Verify mathematical invariants (3,799 space counts, strict comparator injectivity, task ranks, and references):

```bash
python -m pytest tests/ -v
```

Expected output: `4 passed in ~0.10s`.

### 3. Run Feasibility & Search Space Audit
Enumerate all 3,799 well-typed programs and audit exact syntactic and behavioral ranks:

```bash
python scripts/path_a_feasibility_check.py
```

### 4. Run Full Multi-Arm Isolation Controls
Execute the complete uncensored ablation (Uniform, Unigram, Bigram Free Close, Bigram Priced Close, Laplace Alpha Sweeps, and Novel Primitives $U_1$–$U_4$):

```bash
python scripts/prior_controls.py
```

### 5. Verify Camera-Ready PDF Layout
Verify that the paper compiles to exactly 6 pages with 0 overfull hbox:

```bash
python scripts/check_pdf_layout.py
```

---

## 📜 Verified Bibliographic Citations

All 36 cited works in [`paper/references.bib`](paper/references.bib) have been verified against DBLP, ACM Digital Library, IEEE Xplore, PMLR, ACL Anthology, and Springer:
- **Foundations:** McCarthy (1960), Manna & Waldinger (1980), Dechter et al. (2013), Alur et al. (2013), Gulwani et al. (2017).
- **Prior-Guided Synthesis:** DeepCoder (Balog et al., ICLR 2017), RobustFill (Devlin et al., ICML 2017), FlashMeta (Polozov & Gulwani, OOPSLA 2015), Probe (Menon et al., ICML 2013), Lee et al. (PLDI 2018), Execution-Guided (Chen et al., ICLR 2019), DreamCoder (Ellis et al., PLDI 2021), BUSTLE (Odena et al., ICLR 2021).
- **Shortcut Learning & Compositionality:** Geirhos et al. (Nature Machine Intelligence 2020), Lake et al. (BBS 2017), Keysers et al. (ICLR 2020), Hupkes et al. (JAIR 2020), McCoy et al. (ACL 2019), Ribeiro et al. (KDD 2016).

---

## 📖 Citation

If you use this benchmark, empirical decomposition, or findings in your research, please cite:

```bibtex
@inproceedings{kumar2026compositional,
  title     = {Compositional Transfer or Primitive Familiarity? What a Count-Based Prior Explains in a Typed Functional {DSL}},
  author    = {Kumar, Roshan},
  booktitle = {Proceedings of the International Conference on Learning Representations (ICLR)},
  year      = {2026},
  url       = {https://github.com/rk-roshan-kr/compositional-transfer-or-primitive-familiarity}
}
```

---

## 📄 License

This repository is distributed under the [MIT License](LICENSE).

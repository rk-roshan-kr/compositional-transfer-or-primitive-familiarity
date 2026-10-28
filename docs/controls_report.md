# Investigation Report: Path A Prior Controls & Causal Attribution

**File:** `docs/reports/path_a_prior_controls_report.md`  
**Date:** 2026-10-26 (Updated 2026-10-27)  
**Status:** COMPLETE / EMPIRICAL REPORT  
**Author:** Research Agent & Scientific Auditor  
**Script:** `scripts/prior_controls.py`  
**Log:** `logs/prior_controls_20261026.txt`  

---

## 1. The Story: From Headline Result to Root-Cause Dissection

### 1.1. The Initial Breakthrough
In the initial pilot (`scripts/prior_pilot.py`), we trained a count-based bigram prior on 22 canonical AST transitions from 10 simple calibration programs (`T01`–`T10`). When evaluated on 10 held-out tasks (`H01`–`H10`) under a fixed execution budget of $K = 500$:
* **Canonical Enumeration (Arm 1):** Solved 7/10 tasks, hitting the 500 cap on all three Depth-3 compositions (`H08`, `H09`, `H10` at syntactic ranks 1,571, 3,293, and 3,237). Mean executions: **224.2** (uncensored full-space mean: **884.3**, median **176.5**).
* **Learned Prior Search (Arm 2):** Solved **10/10 tasks (100%)** with an average of only **57.1 executions** (median **24.5**), reaching `H08` at 244, `H09` at 79, and `H10` at 117.

The tempting headline conclusion was: *"The model has learned compositional transfer—synthesizing complex Depth-3 compositions by learning abstract grammar rules from Depth-1 and Depth-2 demonstrations."*

### 1.2. The Scientific Challenge
The supervisor audit rejected the claim as unproven, identifying four major confounders:
1. **Search Engine vs. Prior:** Arm 1 was standard Canonical BFS; Arm 2 was Best-First Search with a length penalty ($\lambda = 0.05$) and a free "close list" action. Could the search engine alone be doing the work?
2. **Close Action Free Bias:** Closing a partial list expression with `xs` pushed the candidate program with priority `score` (cost 0), whereas expanding cost `score + lp - \lambda` ($lp < 0$). This structurally favored shallow completions regardless of prior probabilities.
3. **Primitive Overlap (Familiarity vs. Composition):** The 10 held-out tasks exclusively used primitives seen during training (`mul2`, `add1`, `add2`, `gt0`, `even`, `ge0`). Did the model learn compositional syntax, or did it merely learn a frequency list of familiar words?
4. **Generalization Standard:** None of the solutions had been validated on the pinned 20-input hidden generalization suite specified in Section 8.2.

### 1.3. The Plain-English Reality & Exact Attribution
To settle this once and for all, we built a comprehensive multi-arm ablation (`scripts/prior_controls.py`) that stripped away each variable one by one, executing both capped ($K=500$) and full-space uncensored ($N=3,799$) runs:
* **The Engine Test (Uniform Prior):** We turned off the prior entirely (uniform distribution $1/|\mathcal{A}_{\text{valid}}(\sigma)|$). The unguided Best-First engine alone solves **9 out of 10 tasks within budget** and **10/10 uncensored**, achieving an uncensored mean of **208.3** (median **76.0**). **The priority queue alone accounts for 81.7% of the mean search reduction and 66.1% of the median reduction** ($884.3 \to 208.3$ mean, $176.5 \to 76.0$ median).
* **The "Word Cloud" Test (Unigram Prior):** We gave the search a type-conditioned frequency counter that ignored grammar and parent context completely. It achieved an uncensored mean of **63.8** (median **25.0**) and matched the full bigram model on `H08` at **step 244**! **Type-conditioned unigram frequency captures 95.6% of the remaining prior guidance speedup on means and 99.0% on medians** ($208.3 \to 63.8$ mean, $76.0 \to 25.0$ median).
* **Structural Context Margin:** The Bigram contextual transition prior adds an incremental margin of **only 6.7 evaluations on means and 0.5 evaluations on medians** (mean **57.1**, median **24.5**).
* **The Leaf-Close Invariance Test:** Penalizing list closing by $-\lambda$ produced **identical execution counts** across all tasks (mean 57.1). This is algebraically invariant because every complete program performs exactly one terminal leaf close with $xs$.
* **The Real Test (Unseen Primitives U01–U04):** When forced to synthesize programs with unseen primitives (`mul3`, `sub1`, `mod3_0`, `odd`), **the learned prior causes severe negative transfer**, running **2.6$\times$ slower than uniform search** (mean **541.0** vs **209.5**). On Depth-3 task $U_4$, learned priors take **1,795 evaluations** vs Uniform **589 evaluations** (**3.0$\times$ slowdown**), and even lag behind unguided BFS (**1,712 evaluations**).
* **The Correctness Test (Hidden Suite):** Every program found passed **20/20 (100%)** on the pinned hidden suite.

---

## 2. Formal Specification of Experimental Controls

### 2.1. Search Engine Architecture
All best-first arms share the identical priority queue discipline and scoring function:
$$\text{Priority}(P) = -\text{Score}(P)$$
For a partial program prefix $\pi$ expanded at slot $\sigma = (\text{parent}, \text{arg\_pos}, \text{slot\_type})$ with action $a \in \mathcal{A}_{\text{valid}}(\sigma)$:
$$\text{Score}(\pi \circ a) = \text{Score}(\pi) + \ln P(a \mid \sigma) - \lambda$$
Where parsimony penalty $\lambda = 0.05$, budget cap $K = 500$, and max expansion cap $X_{\max} = 100,000$. Search runs uncensored across all $N = 3,799$ candidates.

### 2.2. Prior Model Formulations

#### 1. Uniform Prior Control (Arm 2a)
Isolates the search engine by assigning equal probability to all type-valid options:
$$\forall a \in \mathcal{A}_{\text{valid}}(\sigma), \quad P_{\text{uniform}}(a \mid \sigma) = \frac{1}{|\mathcal{A}_{\text{valid}}(\sigma)|}, \quad \ln P_{\text{uniform}}(a \mid \sigma) = -\ln |\mathcal{A}_{\text{valid}}(\sigma)|$$

#### 2. Type-Conditioned Unigram Marginal Prior Control (Arm 2b)
Isolates primitive frequency from grammatical context while guaranteeing type-valid expansions:
$$P_{\text{unigram}}(a \mid \sigma) = \frac{c(a) + \alpha}{\sum_{a' \in \mathcal{A}_{\text{valid}}(\sigma)} (c(a') + \alpha)}, \quad \alpha = 1.0$$
Context $(\text{parent}, \text{arg\_pos})$ is strictly ignored. In this typed DSL, `ScalarOp`, `Predicate`, and `Reducer` slots belong to unique parent combinators (`map`, `filter`, `fold`), meaning type validity alone enforces parent-combinator matching by construction.

#### 3. Smoothed Bigram Prior (Arms 2c and 2d)
Conditioned on full structural slot context $\sigma = (\text{parent}, \text{arg\_pos}, \text{slot\_type})$ with hierarchical backoff:
$$P_{\text{bigram}}(a \mid \sigma) = \begin{cases}
\frac{c(\sigma, a) + \alpha}{\sum_{a'} (c(\sigma, a') + \alpha)} & \text{if } \sum_{a'} c(\sigma, a') > 0 \quad (\text{Level 1: Full Context}) \\
\frac{c(\text{slot}, a) + \alpha}{\sum_{a'} (c(\text{slot}, a') + \alpha)} & \text{elif } \sum_{a'} c(\text{slot}, a') > 0 \quad (\text{Level 2: Slot Type Alone}) \\
\frac{1}{|\mathcal{A}_{\text{valid}}(\sigma)|} & \text{otherwise} \quad (\text{Level 3: Uniform Fallback})
\end{cases}$$

---

## 3. Empirical Results

### 3.1. Held-Out Evaluation Benchmark (H01–H10 Side-by-Side)

| Task Name | Depth | Enum (Arm 1) | Uniform Prior (Arm 2a) | Unigram Prior (Arm 2b) | Bigram Free (Arm 2c) | Bigram Priced (Arm 2d) | Hidden Suite Pass |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `H01_filter_odds` | 1 | 15 | 8 | 11 | 11 | 11 | 20 / 20 (100%) |
| `H02_double_evens` | 2 | 115 | 171 | 32 | 32 | 32 | 20 / 20 (100%) |
| `H03_add_two_positives` | 2 | 70 | 131 | 30 | 30 | 30 | 20 / 20 (100%) |
| `H04_count_evens` | 2 | 31 | 6 | 7 | 6 | 6 | 20 / 20 (100%) |
| `H05_count_odds` | 2 | 32 | 11 | 20 | 15 | 15 | 20 / 20 (100%) |
| `H06_sum_evens` | 2 | 241 | 20 | 13 | 19 | 19 | 20 / 20 (100%) |
| `H07_sum_positives` | 2 | 238 | 21 | 11 | 18 | 18 | 20 / 20 (100%) |
| `H08_double_positive_evens`| 3 | 500* (1,571) | 500* (955) | **244** | **244** | **244** | 20 / 20 (100%) |
| `H09_sum_doubled_evens` | 3 | 500* (3,293) | 412 | 113 | 79 | 79 | 20 / 20 (100%) |
| `H10_sum_add1_evens` | 3 | 500* (3,237) | 348 | 157 | 117 | 117 | 20 / 20 (100%) |
| **Solved ($K \le 500$)** | — | **7 / 10** | **9 / 10** | **10 / 10** | **10 / 10** | **10 / 10** | **100% Verified** |
| **Mean ($K \le 500$)** | — | **224.2** | **162.8** | **63.8** | **57.1** | **57.1** | — |
| **Uncensored Mean ($N=3799$)** | — | **884.3** | **208.3** | **63.8** | **57.1** | **57.1** | — |
| **Uncensored Median ($N=3799$)** | — | **176.5** | **76.0** | **25.0** | **24.5** | **24.5** | — |

### 3.2. Unseen Primitives Benchmark (U01–U04)

| Task Name | Depth | Primitives Used | Canonical Enum | Uniform Prior | Unigram Prior | Bigram Prior | Hidden Suite |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| `U01_triple_odds` | 2 | `mul3`, `odd` | 130 | **112** | 186 | 186 (slower) | 20 / 20 |
| `U02_sub1_evens` | 2 | `sub1` | **101** | 113 | 144 | 144 (slower) | 20 / 20 |
| `U03_sum_mod3` | 2 | `mod3_0` | 243 | **24** | 31 | 39 (slower) | 20 / 20 |
| `U04_triple_sub1_odds` | 3 | `mul3`, `sub1`, `odd`| 500* (1,712) | 500* (**589**) | 500* (1,795) | 500* (1,795) | 20 / 20* |
| **Mean ($K \le 500$)** | — | — | **243.5** | **187.2** | **215.2** | **217.2** | — |
| **Uncensored Mean ($N=3799$)** | — | — | **546.5** | **209.5** | **539.0** | **541.0** | — |
| **Uncensored Median ($N=3799$)**| — | — | **186.5** | **112.5** | **165.0** | **165.0** | — |

### 3.3. Laplace Smoothing Alpha Sweep ($U_1$–$U_4$)

| Alpha ($\alpha$) | Bigram $U_1$–$U_4$ Executions | Bigram Mean | Unigram Mean | Uniform Mean |
| :--- | :--- | :---: | :---: | :---: |
| 0.01 | [1258, 360, 241, 1795] | 913.5 | 878.0 | **209.5** |
| 0.05 | [1258, 360, 241, 1763] | 905.5 | 870.0 | **209.5** |
| 0.10 | [1258, 328, 225, 1763] | 893.5 | 862.0 | **209.5** |
| 0.20 | [874, 296, 131, 1763] | 766.0 | 751.0 | **209.5** |
| 0.50 | [330, 120, 67, 2071] | 647.0 | 638.0 | **209.5** |
| 1.00 | [186, 144, 39, 1795] | 541.0 | 539.0 | **209.5** |
| 2.00 | [178, 144, 37, 1795] | 538.5 | 536.0 | **209.5** |
| 5.00 | [178, 144, 39, 1795] | 539.0 | 537.0 | **209.5** |
| 10.00 | [178, 144, 37, 1795] | 538.5 | 536.0 | **209.5** |

*Conclusion:* Negative transfer persists across all $\alpha \in [0.01, 10.0]$. Small smoothing parameter ($\alpha \le 0.1$) worsens negative transfer up to 913.5 evaluations ($4.3\times$ slower than Uniform); large smoothing ($\alpha \ge 2.0$) plateaus at 538.5 evaluations ($2.6\times$ slower than Uniform).

### 3.4. 50-Seed Random Tie-Break & Primitive Permutation Sensitivity
* **Uniform Mean across 50 seeds:** $227.1 \pm 55.9$ (min 133.8, max 394.5)
* **Uniform Median across 50 seeds:** $61.9 \pm 22.4$ (min 19.5, max 125.0)

---

## 4. Exact Causal Decomposition Summary

```
--- MEAN-BASED ATTRIBUTION ---
Total BFS -> Bigram Search Reduction: 827.2 evaluations
  1. Priority Queue Mechanics alone (BFS -> Uniform): 676.0 eval (81.7% of total reduction)
  2. Added Prior Guidance Total (Uniform -> Bigram): 151.2 eval
     - Captured by Type-Conditioned Unigram: 144.5 eval (95.6% of prior speedup)
     - Added by Bigram Transition Context: 6.7 eval (4.4% of prior speedup, only 6.7 evals)

--- MEDIAN-BASED ATTRIBUTION ---
Total BFS -> Bigram Search Reduction: 152.0 evaluations
  1. Priority Queue Mechanics alone (BFS -> Uniform): 100.5 eval (66.1% of total reduction)
  2. Added Prior Guidance Total (Uniform -> Bigram): 51.5 eval
     - Captured by Type-Conditioned Unigram: 51.0 eval (99.0% of prior speedup)
     - Added by Bigram Transition Context: 0.5 eval (1.0% of prior speedup, only 0.5 evals)
```

---

## 5. Methodological Recommendations for Future Benchmarks

1. **Mandatory Unigram Baselines:** Any future neurosymbolic benchmark claiming "deep representation learning" or "compositional transfer" must report a smoothed, type-conditioned unigram baseline. If marginal frequency matching explains $\ge 80\%$ of the observed speedup, no compositional claim can be made.
2. **Mandatory Search Engine Parity Controls:** Guidance ablations must compare against the same search engine with a uniform prior, not against naive BFS enumeration.
3. **Disjoint Partitioned-Vocabulary Evaluation:** Benchmarks must evaluate on tasks constructed from primitive subsets disjoint from training demonstrations to confirm whether structural templates generalize independently of lexical familiarity.

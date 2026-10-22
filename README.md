# FlyRank ML Internship — Starter Repo

**Applied Search Intelligence: Google Search Ranking & Discoverability**

This is the starting point for the FlyRank ML Internship. You **clone it into your own public
repo** (one click — *Use this template*), build everything there, and submit that repo URL on
each assignment in your portal — it's your workspace, your submission, and your portfolio all
at once. The rhythm is simple: do the work, commit it, submit on the card. Done.

Everything here runs on a small **anonymized** slice of real FlyRank search data. No credentials,
no private client data, no setup headaches.

> **New here?** Two reads: **[SETUP.md](SETUP.md)** (GitHub, Colab, and data access — ten
> minutes, with every silent pitfall flagged), then **[GUIDE.md](GUIDE.md)** (every file
> explained, what to edit vs. leave alone, and where your own work goes — five minutes).

---

## Project Notebooks — 3-Notebook Modular Architecture

The project provides three template notebooks in [`work/notebooks/`](file:///c:/Users/Matienzo/Desktop/DA-Project/work/notebooks) structured with rich markdown word guides, hypothesis framing, and clear `# TODO` code cells for independent experimentation:

| Step | Notebook | Description | Open in Colab |
|---|---|---|---|
| 1 | `01_problem_framing_and_data_audit` | Problem framing & operational decision, 4 core SEO hypothesis audits, data ingestion scope & limitations, feature vector assembly, and strict anti-leakage defense. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JamesIvanMatienzo/DA-Project/blob/main/work/notebooks/01_problem_framing_and_data_audit.ipynb?flush_cache=true) |
| 2 | `02_model_training` | Grouped holdout client split (zero client domain leakage), transparent heuristic baselines (Mean Dummy & Word Count Rule), and candidate model training guide. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JamesIvanMatienzo/DA-Project/blob/main/work/notebooks/02_model_training.ipynb?flush_cache=true) |
| 3 | `03_evaluation_and_conclusions` | Official benchmark comparisons ($R^2$, RMSE, MAE, Spearman $\rho$), diagnostic residual plots, sub-cohort analysis, outlier post-mortems, and operational refresh queue export. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JamesIvanMatienzo/DA-Project/blob/main/work/notebooks/03_evaluation_and_conclusions.ipynb?flush_cache=true) |

### Running Locally

```bash
git clone https://github.com/JamesIvanMatienzo/DA-Project.git
cd DA-Project
pip install -r requirements.txt
jupyter notebook work/notebooks/
```



---

## What you get

| Path | What it is |
|---|---|
| `data/raw/content_refresh_anonymized.csv` | The anonymized starter dataset (~30k pages). |
| `outputs/` | Example outputs so you can see the **target shape** (`model_report.md`, `refresh_queue_sample.csv`, `charts/`). |
| `work/` | **Your space.** Lane experiments and your capstone live here — see `work/README.md`. |
| `docs/` | The core docs + the data dictionary (see below). |

### Read these (in `docs/`)

1. **`ml-core-foundation-framework.md`** — the first-principles map of ML as a whole system. The backbone of the live sessions.
2. **`ml-intern-dataset-and-lane-guide.md`** — how to use the data safely, the capstone workflow, and the analysis "lanes" you can pick from.
3. **`intern-free-tooling-guide.md`** — the zero-budget tool stack (Python, Colab, free AI assistants). You never need to pay for anything.
4. **`data-dictionary.md`** — all 44 columns: meaning, scale, and gotchas. Keep it open while you work.

---

## The pipeline (what `run_all.py` does)

```text
01_prepare_features.py   clean + build the feature vector, define the label
02_baseline_score.py     a transparent hand-rule "fix this first" score
03_train_model.py        logistic regression, decision tree, random forest (client-holdout split)
04_evaluate_and_export.py  ranked queue + charts + Markdown report
05_build_pdf_report.py   a shareable PDF summary
```

On the bundled sample, the learned model clearly beats the hand-written rule at picking the right
pages to review first (**Precision@50 ≈ 0.24 → 0.74**; the model number can land 0.68–0.74
depending on library versions — the ~3x lift is the point). The notebooks compute these numbers
live, so they always reflect the current data and environment.

**Teaching point:** the model is the capstone, but the *workflow* is the lesson —
`problem framing → data cleaning → baseline → first model → evaluation → explainable recommendation`.

---

## Data safety (read `DATA_USE.md`)

- Only the small **anonymized** CSV ships here — no client names, domains, URLs, titles, or keywords.
- **Never** add raw private client data to this repo or your fork. Need more data? Request an approved
  release from your mentor — never export it yourself.
- Don't paste client data into third-party AI tools.
- Frame every result as **observed / measured / directional / decision-support** — never
  "I predicted Google's algorithm."

The `.gitignore` blocks datasets by default, and CI fails any commit that includes a dataset.

---

## Assignments & schedule

Weekly assignments, live events, and the capstone live on **your portal board** (your
enrollment email has your access link). This repo is the shared technical foundation they all
build on — and the `skills/` folder here is the instruction library for your AI assistant
(start at [skills/README.md](skills/README.md)).

**First time with GitHub?** You need exactly four things (full walkthrough: [SETUP.md](SETUP.md)):
1. A free account at github.com.
2. Your own copy of this repo: **Use this template → Create a new repository** → public.
   (One click — brings the notebooks, `work/`, and the CI leak-guard with it.)
3. In Colab: *File → Save a copy in GitHub* — opened from your copy's badges, the dialog is
   already pre-filled with your repo and path, so it's just OK (Colab handles auth).
4. That's your submission repo — share its **github.com/you/your-repo** URL with Assignment 1
   (never a colab.research.google.com or drive.google.com link).

---

*Track leads: Mirza Ašćerić (ML) · Hole (data engineering). Code under MIT (see `LICENSE`); data under `DATA_USE.md`.*

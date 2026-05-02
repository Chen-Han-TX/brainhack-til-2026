# BrainHack TIL-AI 2026 — Team Onboarding

## Welcome

We're competing in **DSTA BrainHack TIL-AI 2026** (Novice track) — a 2-week online
qualifier (May 9–23) where we submit Dockerised ML models against defence-AI
challenges. Role split: **Yuxuan → Computer Vision (object detection)**,
**Eugene → OCR (text recognition)**, **Chen Han → team lead + infra**. You have
**~10 days until Training Day on May 9**. This doc walks you from a clean laptop
to "ready to run a HuggingFace model on GCP." Don't skip sections — each one
unblocks the next.

---

## Section 1 — Local Python environment (30 min)

We use `venv` so each project's packages stay isolated (no global mess).

- [ ] Install Python 3.10+ from <https://www.python.org/downloads/>
- [ ] Open a terminal in your projects folder and run:

```bash
# Mac / Linux
python3 -m venv .venv && source .venv/bin/activate

# Windows (PowerShell)
python -m venv .venv ; .venv\Scripts\Activate.ps1
```

- [ ] Verify: `python --version` should print `3.10.x` or higher, and your prompt
  should now start with `(.venv)`.

> **If `python3` not found** on Mac: run `xcode-select --install` first.

---

## Section 2 — Docker (45 min)

BrainHack scoring works by you submitting a **Docker image** to their platform.
The grader spins up your container, sends inputs, scores outputs. So our model
+ inference code must run inside a container — Docker lets us match that
environment locally before submitting.

- [ ] Install Docker Desktop: [Mac](https://docs.docker.com/desktop/install/mac-install/) · [Windows](https://docs.docker.com/desktop/install/windows-install/)
- [ ] **Launch Docker Desktop** (the whale icon must be in your menu bar — commands
  silently fail if the daemon isn't running)
- [ ] Verify:

```bash
docker run hello-world
```

You should see "Hello from Docker!". If you get `Cannot connect to the Docker
daemon`, Docker Desktop isn't running yet — open the app and wait ~30s.

---

## Section 3 — GCP & Vertex AI Workbench (1 hour)

DSTA runs the official training environment on **Google Cloud Platform**.
We'll mirror that locally on Workbench (managed JupyterLab) so what works for
you works for the grader.

- [ ] Sign up at <https://console.cloud.google.com> with your **NTU email**
  (free $300 trial credit). Region: **asia-southeast1 (Singapore)**.
- [ ] **Set a billing budget BEFORE doing anything else.** GPU instances cost
  ~SGD 1–3/hour; one forgotten weekend = your trial gone.
  - Billing → Budgets & alerts → **Create budget: $50, alerts at 50% / 90% / 100%**
- [ ] Enable the **Vertex AI API** (Console → APIs & Services → Enable APIs).

> **Naming:** Google rebranded "Vertex AI" → "Vertex AI Agent Platform" in late
> 2025. The product is the same. If you can't find it, search "**Workbench**".

- [ ] Create a Workbench instance: Vertex AI → Workbench → **Create instance**
  - **For now (testing):** machine type `e2-standard-4` (CPU only, ~SGD 0.20/hr)
  - **Later (real training):** `n1-standard-4` + 1× **NVIDIA T4** GPU (~SGD 1/hr)
- [ ] Open JupyterLab → **File → New → Terminal** and verify:

```bash
nvidia-smi          # skip on CPU instance — only on GPU
docker --version
docker run hello-world
python3 --version
git --version
```

- [ ] **STOP the instance when done.** Workbench → select instance → **Stop**.
  An idle running instance still bills you. Treat it like a taxi meter.

---

## Section 4 — Git / GitHub (15 min)

We work in a **private repo**: `<REPO_URL_TBD>`.

- [ ] Send Chen Han your GitHub username so he can add you as a collaborator
- [ ] Clone:

```bash
git clone <REPO_URL_TBD> && cd brainhack-til-2026
```

**Branching strategy:**
- `main` is always stable — never push directly
- Feature work goes on `feat/<your-name>-<short-desc>` (e.g. `feat/eugene-trocr-baseline`)
- Open a PR into `main` when ready

**Eugene** — read [Pro Git ch. 1–3](https://git-scm.com/book/en/v2) before May 9
(branches, merging, remotes). **Yuxuan** — you're set, just follow the branch
naming.

---

## Section 5 — Run your first HuggingFace inference (30 min)

This is the litmus test. If this works, your stack is ready.

- [ ] In your activated venv:

```bash
pip install transformers torch
```

- [ ] Save as `hf_test.py` and run `python hf_test.py`:

```python
from transformers import pipeline
clf = pipeline("sentiment-analysis")
print(clf("BrainHack 2026 is going to be fun!"))
```

Expected: `[{'label': 'POSITIVE', 'score': 0.99...}]`. The first run downloads
the model (~250 MB, cached locally). `pipeline()` hides the tokenizer + model
+ post-processing — for the real challenges you'll wire those up yourself, but
this proves the plumbing works.

---

## Section 6 — Repo tour

```
brainhack-til-2026/
├── CLAUDE.md            # Project context (read this!)
├── README.md            # Image-classification template usage
├── configs/             # YAML hyperparameter configs
├── src/                 # Training / eval / data / models
├── experiments/         # Per-track warmup work — START HERE
│   └── asr_warmup/      # Chen Han's warmup (ASR exploration)
└── docs/                # This file lives here
```

`experiments/<track>_warmup/` is your sandbox. You'll create
**`experiments/cv_warmup/`** (Yuxuan) and **`experiments/ocr_warmup/`** (Eugene)
to hold tutorial code, notes, and a small README documenting what you tried.

---

## Section 7 — Your specific next steps

### Yuxuan — Computer Vision (object detection)

- [ ] Read [YOLOv8 quickstart](https://docs.ultralytics.com/quickstart/)
- [ ] Run the 5-line YOLOv8 demo on a sample image; commit to `experiments/cv_warmup/`
- [ ] Skim [COCO dataset format](https://cocodataset.org/#format-data) (most detection challenges use it)
- [ ] **Bring to Training Day:** laptop, charger, working venv + Docker, GCP account active

### Eugene — OCR (text recognition)

- [ ] Read [TrOCR quickstart](https://huggingface.co/docs/transformers/model_doc/trocr)
- [ ] Run the TrOCR HuggingFace example on a sample text image; commit to `experiments/ocr_warmup/`
- [ ] Skim [an intro to CTC loss](https://distill.pub/2017/ctc/) — this is how OCR/ASR models align outputs
- [ ] **Bring to Training Day:** laptop, charger, working venv + Docker, GCP account active

---

## Section 8 — Communication & help

- **Group chat** for status, questions anyone could answer, celebration
- **DM Chen Han** for blockers, GCP/billing issues, anything sensitive
- **30-minute rule:** stuck for >30 min? Ask. We have ~10 days, don't burn a day on a typo.
- **Daily standup:** one line in chat — "yesterday / today / blocked"
- **Weekly sync:** TBD (Chen Han to confirm based on shift schedule)

---

## Section 9 — Resources

- BrainHack official: <https://www.dsta.gov.sg/brainhack>
- Last year's template: <https://github.com/til-ai/til-25>
- HuggingFace docs: <https://huggingface.co/docs>
- Claude.ai (great for "explain this error" / "why does this code do X") — Chen Han uses it heavily, you should too
- Project context for AI assistants: see `CLAUDE.md` in repo root

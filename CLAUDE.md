# CLAUDE.md — BrainHack TIL-AI 2026

## Project Overview

This is a competition project for **DSTA BrainHack 2026 — TIL-AI (Today I Learned - AI)** track, Novice category. BrainHack is an annual AI/defence tech competition organized by Singapore's Defence Science and Technology Agency (DSTA).

**Team:** 3x NTU Computer Science Year 1 students (Novice track)
**Timeline:** Qualifiers May 9–23 (online), Finals Jun 10–11 (in-person, if qualified)
**Goal:** Build and deploy ML models for defence-inspired AI challenges. Qualify for finals and aim for Top 3 Novice.

## Competition Format

- **Qualifiers (May 9–23):** Virtual. Submit ML models to an online platform. Best-performing result counts. Can submit anytime during the 2-week window. Multiple submissions allowed — iterate fast.
- **Finals (Jun 10–11):** In-person at DSTA. Finalists compete by completing AI challenges and collecting points in a physical arena. Expect curveball tasks not seen during qualifiers.
- **Past year tasks have included:** image classification, object detection, automatic speech recognition (ASR), optical character recognition (OCR), reinforcement learning. Computer vision tasks appear every year.

## Tech Stack

- **Language:** Python 3.10+
- **Deep Learning:** PyTorch, torchvision
- **ML Utilities:** scikit-learn, numpy, pandas, matplotlib
- **Pretrained Models:** HuggingFace transformers, torchvision model zoo
- **Experiment Tracking:** (TBD — consider Weights & Biases or simple CSV logging)
- **Deployment:** May need Docker/TensorRT for finals
- **Compute:** Local machines + Google Colab (free GPU) as backup

## Current Project Structure

```
brainhack-til-2026/
├── CLAUDE.md                        # This file — project context for Claude
├── README.md                        # Usage guide for the template
├── requirements.txt                 # Pinned Python dependencies
├── .gitignore                       # Excludes data/, checkpoints/, runs/, etc.
│
├── configs/
│   └── default.yaml                 # Master config: all hyperparameters live here
│                                    # Covers: experiment, data, model, optim,
│                                    # scheduler, train, eval sections.
│                                    # CLI dotted overrides work without editing YAML.
│
└── src/
    ├── __init__.py
    │
    ├── data/
    │   ├── __init__.py
    │   └── datasets.py              # Dataset registry + dataloader factory.
    │                                # Register new datasets with @register_dataset.
    │                                # Built-ins: cifar10, cifar100, imagefolder.
    │                                # Augmentation pipeline (train): RandomCrop/
    │                                # RandomResizedCrop, RandomHorizontalFlip,
    │                                # RandomRotation, ColorJitter, Normalize.
    │                                # Eval pipeline: Resize → CenterCrop → Normalize.
    │                                # All augmentation knobs read from config YAML.
    │
    ├── models/
    │   ├── __init__.py
    │   └── registry.py              # Model registry + builder helpers.
    │                                # Register new models with @register_model.
    │                                # Built-ins: resnet18 (default), resnet34,
    │                                # resnet50, mobilenet_v3_small, efficientnet_b0.
    │                                # Auto-replaces classifier head for num_classes.
    │                                # small_input_stem: true swaps ResNet 7x7 stem
    │                                # for 3x3 (needed for CIFAR-sized inputs).
    │
    ├── utils/
    │   ├── __init__.py
    │   ├── config.py                # YAML loader returning attribute-access Config.
    │                                # Supports CLI dotted overrides:
    │                                #   load_config("cfg.yaml", ["train.epochs=10"])
    │   ├── metrics.py               # AverageMeter (running avg) + top-k accuracy().
    │   └── seed.py                  # set_seed(seed, deterministic=False) for
    │                                # reproducible runs across Python/NumPy/PyTorch.
    │
    ├── train.py                     # Training entrypoint.
    │                                # python -m src.train --cfg configs/default.yaml
    │                                # Features: AMP (mixed precision), grad clipping,
    │                                # cosine/step/none LR schedulers, tqdm progress,
    │                                # JSONL history log, last.pt + best.pt checkpoints.
    │
    └── evaluate.py                  # Evaluation entrypoint.
                                     # python -m src.evaluate --cfg configs/default.yaml \
                                     #   --ckpt runs/exp/best.pt
                                     # Prints loss, top-1, top-5 accuracy as JSON.
```

## Quick Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Train (defaults: ResNet18 pretrained, CIFAR-10, 50 epochs, SGD + cosine LR)
python -m src.train --cfg configs/default.yaml

# Train with inline overrides (no YAML edit needed)
python -m src.train --cfg configs/default.yaml \
    model.name=resnet50 train.epochs=10 optim.lr=0.05 data.batch_size=64

# Evaluate a checkpoint
python -m src.evaluate --cfg configs/default.yaml \
    --ckpt runs/resnet18_cifar10/best.pt

# Swap to ImageNet-style data (folder layout: root/train/<class>/* root/val/<class>/*)
# Set in YAML: data.name=imagefolder, data.image_size=224, model.small_input_stem=false
```

## How to Extend

**New dataset:** Add `@register_dataset("name")` builder in `src/data/datasets.py` returning `(train_dataset, val_dataset)`. The augmentation pipeline is shared — set knobs in YAML.

**New model:** Add `@register_model("name")` builder in `src/models/registry.py` returning an `nn.Module`. Then set `model.name: name` in the YAML.

**New config:** Copy `configs/default.yaml`, change what you need, pass with `--cfg`.

## Team Context

- **Chen Han (team lead):** Has prior ML/DL coursework from polytechnic (Ngee Ann Polytechnic — Machine Learning B+, Deep Learning A). Built an NLP chatbot during internship at AIDA Technologies (2022). Currently NTU CS Y1. Strongest ML background on the team but knowledge is rusty (last touched ML in 2022). Currently doing National Service reservist (shift work) so has limited but consistent time blocks.
- **Teammate 2 & 3:** NTU CS Y1 students. No prior AI/ML experience. Can code in Python. Learning ML basics from scratch during prep period.

## Key Constraints

- Team members have **limited GPU access**. Chen Han's local machine is a MacBook Pro M4 Pro (24 GB unified memory) — training uses Apple Silicon **MPS** (auto-selected by `train.py` / `evaluate.py` when CUDA is absent). Larger / longer runs go to Google Colab or Kaggle notebooks.
- AMP is **CUDA-only** in this codebase (`torch.cuda.amp.GradScaler`); MPS and CPU runs are fp32. The `train.amp: true` config flag does nothing off-CUDA.
- Chen Han is on **shift work** (2 morning / 2 afternoon / 2 off rotation) so work happens in fragmented time blocks, not marathon sessions.
- Team is **learning while competing** — code should be clean, well-commented, and easy to understand. Avoid unnecessarily complex architectures.

## Development Principles

1. **Start simple, iterate fast.** Begin with a pretrained model (ResNet18/EfficientNet-B0) + basic augmentation. Get a baseline score, then improve incrementally.
2. **Pretrained models over training from scratch.** Transfer learning is the winning strategy for Novice teams with limited compute and data.
3. **Data augmentation is critical.** Past TIL-AI winners consistently cite aggressive augmentation as a key factor. Use torchvision transforms and albumentations.
4. **Document every experiment.** Log what you tried, what the result was, and why. This helps the team avoid repeating failed approaches and helps write the post-competition reflection.
5. **Code must be modular.** The competition may throw different task types (CV, ASR, NLP). Modular code lets us swap datasets and models quickly.
6. **Clean commits.** Use descriptive commit messages. This repo will become a GitHub portfolio piece after the competition.

## Past TIL-AI References

- TIL 2023 Champion: https://github.com/aliencaocao/TIL-2023
- TIL 2025 Template: https://github.com/til-ai/til-25
- TIL 2019 1st Place Writeup: https://medium.com/@pyesonekyaw/1st-place-solution-for-dsta-brainhack-2019-today-i-learned-ai-challenge-f46689b44a73

## Common Commands

```bash
# Train a model
python src/train.py --model resnet18 --epochs 30 --lr 0.001

# Evaluate on validation set
python src/evaluate.py --checkpoint checkpoints/best_model.pth

# Generate submission
python src/predict.py --checkpoint checkpoints/best_model.pth --output submissions/

# Run Jupyter notebook
jupyter notebook notebooks/
```

## What Claude Should Know When Helping

- Always suggest the **simplest working solution first**, then offer optimizations.
- When suggesting models, prefer **torchvision pretrained models** (ResNet, EfficientNet, MobileNet) over exotic architectures.
- Explain ML concepts in plain language — not all team members have ML background.
- When writing training code, always include: **validation split, early stopping, model checkpointing, training/validation loss logging**.
- Flag potential **overfitting** aggressively — with small competition datasets this is the #1 failure mode.
- For data augmentation, default to: RandomHorizontalFlip, RandomRotation(15), ColorJitter, RandomResizedCrop, Normalize with ImageNet stats.
- Code should include **type hints** and **docstrings** for readability.
- Use **argparse or a config dict** for hyperparameters — never hardcode values in training loops.

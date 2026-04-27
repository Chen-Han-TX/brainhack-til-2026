# PyTorch Image Classification Template

A minimal, swappable PyTorch image classification project. Configured by YAML,
defaults to a torchvision-pretrained **ResNet18** on **CIFAR-10**, and is
structured so you can drop in new datasets or models with a single decorator.

## Layout

```
.
├── configs/
│   └── default.yaml          # All hyperparameters live here
└── src/
    ├── data/datasets.py      # Dataset registry + augmentation pipeline
    ├── models/registry.py    # Model registry (ResNet18 default)
    ├── utils/                # config / seed / metrics
    ├── train.py              # Training entrypoint
    └── evaluate.py           # Evaluation entrypoint
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Train

```bash
python -m src.train --cfg configs/default.yaml
```

Override any field on the command line with dotted keys:

```bash
python -m src.train --cfg configs/default.yaml \
    train.epochs=10 optim.lr=0.05 data.batch_size=64
```

Outputs (checkpoints, resolved config, JSONL training history) are written
to `cfg.experiment.output_dir` (default: `./runs/resnet18_cifar10`).

## Evaluate

```bash
python -m src.evaluate --cfg configs/default.yaml --ckpt runs/resnet18_cifar10/best.pt
```

## Adding a new dataset

```python
# src/data/datasets.py
@register_dataset("my_dataset")
def _build_my_dataset(data_cfg):
    train = MyDataset(..., transform=_build_train_transform(data_cfg))
    val   = MyDataset(..., transform=_build_eval_transform(data_cfg))
    return train, val
```

Then set `data.name: my_dataset` in the YAML config. Augmentation knobs
(`horizontal_flip`, `rotation_degrees`, `color_jitter`, `random_crop_padding`)
are honored automatically through the shared transform builders.

## Adding a new model

```python
# src/models/registry.py
@register_model("my_net")
def _my_net(model_cfg, num_classes):
    return MyNet(num_classes=num_classes, pretrained=model_cfg.get("pretrained", False))
```

Then set `model.name: my_net` in the YAML config. Built-in registry entries:
`resnet18`, `resnet34`, `resnet50`, `mobilenet_v3_small`, `efficientnet_b0`.

## Notes

- `model.small_input_stem: true` swaps ResNet's 7x7/stride-2 stem for a 3x3
  stride-1 stem and removes the initial maxpool. This is the canonical fix
  for training ResNets on CIFAR-sized inputs and should be **disabled** for
  ImageNet-resolution data.
- Device is auto-selected: CUDA → MPS (Apple Silicon) → CPU.
- AMP (`train.amp: true`) only activates on CUDA. MPS and CPU runs are fp32 —
  `torch.cuda.amp.GradScaler` is CUDA-only and the config flag is ignored
  off-CUDA.
- Determinism: set `experiment.seed` and pass `deterministic=True` to
  `set_seed` if you need bit-exact runs (slower).

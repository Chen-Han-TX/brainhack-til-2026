"""Model registry.

Register a new architecture by decorating a builder with
``@register_model("my_net")``. The builder receives the parsed model config
and the dataset-derived ``num_classes`` and must return an ``nn.Module`` whose
final layer matches ``num_classes``.
"""

from __future__ import annotations

from typing import Callable, Mapping

import torch.nn as nn
from torchvision import models as tvm


ModelBuilder = Callable[[Mapping, int], nn.Module]
_MODEL_REGISTRY: dict[str, ModelBuilder] = {}


def register_model(name: str) -> Callable[[ModelBuilder], ModelBuilder]:
    """Decorator to register a model builder under a string key."""

    def decorator(fn: ModelBuilder) -> ModelBuilder:
        key = name.lower()
        if key in _MODEL_REGISTRY:
            raise ValueError(f"Model {name!r} already registered")
        _MODEL_REGISTRY[key] = fn
        return fn

    return decorator


def list_models() -> list[str]:
    return sorted(_MODEL_REGISTRY)


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #


def _adapt_resnet_stem_for_small_images(model: nn.Module) -> None:
    """Replace ResNet's 7x7/stride-2 stem + maxpool with a 3x3/stride-1 stem.

    Standard torchvision ResNets downsample 4x in the stem, which destroys
    resolution on 32x32 inputs (e.g. CIFAR). Swapping to a 3x3 stride-1 conv
    and removing the initial maxpool is the canonical fix for small images.
    """
    in_channels = model.conv1.in_channels
    out_channels = model.conv1.out_channels
    model.conv1 = nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
    )
    model.maxpool = nn.Identity()


def _replace_classifier(model: nn.Module, num_classes: int) -> nn.Module:
    """Swap the final classification head to match ``num_classes``.

    Works for the most common torchvision architectures. Extend as needed.
    """
    # ResNet / ResNeXt / RegNet expose `fc`.
    if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model

    # MobileNet / EfficientNet expose `classifier` as Sequential ending in Linear.
    if hasattr(model, "classifier"):
        classifier = model.classifier
        if isinstance(classifier, nn.Sequential) and isinstance(
            classifier[-1], nn.Linear
        ):
            in_features = classifier[-1].in_features
            classifier[-1] = nn.Linear(in_features, num_classes)
            return model
        if isinstance(classifier, nn.Linear):
            in_features = classifier.in_features
            model.classifier = nn.Linear(in_features, num_classes)
            return model

    raise NotImplementedError(
        f"Don't know how to replace the classifier of {type(model).__name__}; "
        "extend _replace_classifier or write a custom builder."
    )


def _build_torchvision(
    factory_name: str,
    weights_enum_name: str,
    model_cfg: Mapping,
    num_classes: int,
) -> nn.Module:
    """Generic builder for torchvision classification models."""
    factory = getattr(tvm, factory_name)
    weights = None
    if model_cfg.get("pretrained", False):
        weights_cls = getattr(tvm, weights_enum_name)
        weights = getattr(weights_cls, "DEFAULT")

    model = factory(weights=weights)

    if model_cfg.get("small_input_stem", False) and factory_name.startswith("resnet"):
        _adapt_resnet_stem_for_small_images(model)

    return _replace_classifier(model, num_classes)


# --------------------------------------------------------------------------- #
# Built-in models                                                             #
# --------------------------------------------------------------------------- #


@register_model("resnet18")
def _resnet18(model_cfg: Mapping, num_classes: int) -> nn.Module:
    return _build_torchvision("resnet18", "ResNet18_Weights", model_cfg, num_classes)


@register_model("resnet34")
def _resnet34(model_cfg: Mapping, num_classes: int) -> nn.Module:
    return _build_torchvision("resnet34", "ResNet34_Weights", model_cfg, num_classes)


@register_model("resnet50")
def _resnet50(model_cfg: Mapping, num_classes: int) -> nn.Module:
    return _build_torchvision("resnet50", "ResNet50_Weights", model_cfg, num_classes)


@register_model("mobilenet_v3_small")
def _mobilenet_v3_small(model_cfg: Mapping, num_classes: int) -> nn.Module:
    return _build_torchvision(
        "mobilenet_v3_small", "MobileNet_V3_Small_Weights", model_cfg, num_classes
    )


@register_model("efficientnet_b0")
def _efficientnet_b0(model_cfg: Mapping, num_classes: int) -> nn.Module:
    return _build_torchvision(
        "efficientnet_b0", "EfficientNet_B0_Weights", model_cfg, num_classes
    )


# --------------------------------------------------------------------------- #
# Public factory                                                              #
# --------------------------------------------------------------------------- #


def build_model(model_cfg: Mapping, num_classes: int) -> nn.Module:
    """Instantiate a model from config.

    Args:
        model_cfg:  ``cfg.model`` mapping (must contain ``name``).
        num_classes: number of output classes (typically derived from data).
    """
    name = str(model_cfg["name"]).lower()
    if name not in _MODEL_REGISTRY:
        raise KeyError(
            f"Unknown model {name!r}. Available: {list_models()}. "
            "Register new models with @register_model."
        )
    return _MODEL_REGISTRY[name](model_cfg, num_classes)

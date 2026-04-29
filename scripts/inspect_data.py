"""Quick script to inspect what build_dataloaders returns.
Run: python -m scripts.inspect_data
"""
from src.utils import load_config
from src.data import build_dataloaders


def main():
    cfg = load_config("configs/default.yaml")
    train_loader, val_loader, num_classes = build_dataloaders(cfg.data)

    print(f"num_classes: {num_classes}")
    print(f"train batches: {len(train_loader)}")
    print(f"val batches: {len(val_loader)}")

    for images, targets in train_loader:
        print(f"images shape: {images.shape}")
        print(f"images dtype: {images.dtype}")
        print(f"images range: [{images.min():.3f}, {images.max():.3f}]")
        print(f"targets shape: {targets.shape}")
        print(f"targets sample: {targets[:10]}")
                
                # 看看一个完整图片的tensor
        print(f"\nFirst image tensor:")
        print(f"  shape: {images[0].shape}")        # [3, 32, 32] —— 单张图，没了N维度
        print(f"  R channel range: [{images[0, 0].min():.3f}, {images[0, 0].max():.3f}]")
        print(f"  G channel range: [{images[0, 1].min():.3f}, {images[0, 1].max():.3f}]")
        print(f"  B channel range: [{images[0, 2].min():.3f}, {images[0, 2].max():.3f}]")

        # 看看label分布
        import torch
        unique, counts = torch.unique(targets, return_counts=True)
        print(f"\nLabel distribution in this batch:")
        for label, count in zip(unique.tolist(), counts.tolist()):
            print(f"  class {label}: {count} samples")
        break



if __name__ == "__main__":
    main()
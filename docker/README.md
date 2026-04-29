# Docker — Inference Image

A CPU-only image that runs `src.inference` against a checkpoint produced by
`src.train`. Designed to mimic the BrainHack TIL-AI submission flow: build an
image that takes a checkpoint + image and prints predictions.

## Build

From the repo root (the build context must be the repo root because the
Dockerfile copies `src/`):

```bash
docker build -t brainhack-til:inference .
```

First build downloads the CPU PyTorch wheels (~200 MB on arm64,
larger on amd64). Subsequent builds reuse the layer cache.

### Cross-arch builds

The default build matches your host arch. On an M-series Mac that's
`linux/arm64`. To produce an `linux/amd64` image (closer to most submission
servers):

```bash
docker buildx build --platform linux/amd64 -t brainhack-til:inference-amd64 .
```

## Run

`runs/` is excluded from the image by `.dockerignore`, so the checkpoint must
be **mounted at runtime**. Same for the input image. The simplest pattern
mounts the entire repo read-only at `/data`:

```bash
docker run --rm \
    -v "$(pwd):/data:ro" \
    brainhack-til:inference \
    --ckpt /data/runs/resnet18_cifar10/best.pt \
    --image /data/path/to/image.jpg
```

Tighter mounts (one volume per file) are equivalent:

```bash
docker run --rm \
    -v "$(pwd)/runs/resnet18_cifar10/best.pt:/ckpt/best.pt:ro" \
    -v "$(pwd)/path/to/image.jpg:/input/image.jpg:ro" \
    brainhack-til:inference \
    --ckpt /ckpt/best.pt \
    --image /input/image.jpg
```

Output is JSON on stdout — top-5 by default, configurable with `--topk N`:

```json
{
    "checkpoint": "/data/runs/resnet18_cifar10/best.pt",
    "image": "/data/path/to/image.jpg",
    "top_k": [
        {"class_index": 3, "probability": 0.42},
        ...
    ]
}
```

CIFAR-10 class index ordering (for reference, not baked into the image):
`0=airplane, 1=automobile, 2=bird, 3=cat, 4=deer, 5=dog, 6=frog, 7=horse, 8=ship, 9=truck`.

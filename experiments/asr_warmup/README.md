# ASR Warmup — Whisper Inference

Warmup experiment for the **BrainHack 2026 TIL-AI ASR challenge**. The goal is to get comfortable loading a pretrained OpenAI Whisper model via HuggingFace Transformers and running inference on a single audio file. No training — just inference + dockerization, mimicking the BrainHack ASR submission pattern.

This experiment is isolated from the CIFAR pipeline at the repo root. Treat it as a self-contained sandbox.

## Layout

```
experiments/asr_warmup/
├── data/audio/           # drop test.m4a (or .wav/.mp3) here — gitignored
├── src/                  # inference.py lives here (written manually)
├── requirements-asr.txt  # pinned deps for this experiment only
├── Dockerfile            # CPU-only image, mirrors BrainHack submission shape
└── .dockerignore
```

## Local setup

From the repo root, with the existing `.venv` activated (or a fresh venv):

```bash
pip install -r experiments/asr_warmup/requirements-asr.txt
```

Drop an audio file into `experiments/asr_warmup/data/audio/test.m4a` (Whisper handles `.m4a`, `.wav`, `.mp3`, `.flac`, etc. via librosa + ffmpeg).

## Running inference

```bash
cd experiments/asr_warmup
python -m src.inference --audio data/audio/test.m4a --model openai/whisper-tiny
```

Suggested model sizes to try (CPU-friendly first):

| Model              | Params | Notes                                  |
|--------------------|--------|----------------------------------------|
| `openai/whisper-tiny`  | 39M  | Fast smoke test, weak accuracy         |
| `openai/whisper-base`  | 74M  | Good CPU baseline                      |
| `openai/whisper-small` | 244M | Best accuracy/speed tradeoff on CPU    |

## Docker

Build (run from inside `experiments/asr_warmup/`):

```bash
docker build -t asr-warmup .
```

Run inference, mounting the local audio folder so we don't bake audio into the image:

```bash
docker run --rm \
    -v "$(pwd)/data/audio:/app/data/audio" \
    asr-warmup --audio /app/data/audio/test.m4a --model openai/whisper-tiny
```

The first run will download the Whisper weights from HuggingFace into the container's HF cache. To persist that cache across runs, also mount `~/.cache/huggingface`:

```bash
docker run --rm \
    -v "$(pwd)/data/audio:/app/data/audio" \
    -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
    asr-warmup --audio /app/data/audio/test.m4a --model openai/whisper-tiny
```

## References

- HuggingFace Whisper docs: https://huggingface.co/docs/transformers/model_doc/whisper
- OpenAI Whisper paper (Radford et al., 2022): https://arxiv.org/abs/2212.04356
- Model cards: https://huggingface.co/openai/whisper-tiny / `whisper-base` / `whisper-small`

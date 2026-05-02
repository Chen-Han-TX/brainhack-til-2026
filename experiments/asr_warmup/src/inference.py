"""Whisper ASR inference — transcribe a single audio file, no training required.

What this script does
---------------------
Loads OpenAI's pretrained Whisper model from HuggingFace and transcribes one
audio file to text. Nothing is trained or fine-tuned here — we're purely using
the weights OpenAI already released.

Whisper's preprocessing pipeline
---------------------------------
1. Audio is resampled to 16 000 Hz and converted to mono float32.
2. WhisperProcessor computes a **log-mel spectrogram** (80 mel bands, 25 ms
   windows, 10 ms hop). This turns raw audio into an image-like 2-D feature
   that the transformer encoder can process.
3. Whisper always processes **30-second chunks**. The processor zero-pads
   shorter clips and — for longer files — you would need to chunk manually
   (this script handles files up to 30 s; anything longer is silently truncated
   by the processor, which is fine for warmup purposes).

Why WhisperProcessor instead of manual preprocessing
------------------------------------------------------
WhisperProcessor bundles the feature extractor (audio → mel spectrogram) and
the tokenizer (token ids → text) in one object. It handles sampling-rate
validation, padding, and return-tensors boilerplate so we don't have to.

Usage
-----
    python -m experiments.asr_warmup.src.inference --audio path/to/audio.m4a
    python -m experiments.asr_warmup.src.inference --audio audio.wav --language fr
"""

from __future__ import annotations

import argparse
import json

import librosa
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------


def _get_device() -> torch.device:
    """Prefer CUDA → Apple MPS → CPU, mirroring the rest of this repo."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Audio loading
# ---------------------------------------------------------------------------

# Whisper was trained exclusively on 16 kHz audio. Feeding a different sample
# rate without resampling would shift every frequency bin in the mel
# spectrogram, causing garbage outputs.
WHISPER_SAMPLE_RATE = 16_000


def load_audio(path: str) -> torch.Tensor:
    """Load an audio file, convert to mono 16 kHz float32 tensor.

    Args:
        path: path to any audio format librosa supports (wav, mp3, m4a, flac…).

    Returns:
        1-D float32 tensor of waveform samples at 16 000 Hz.
    """
    # librosa.load returns (waveform: np.ndarray, sample_rate: int).
    # mono=True collapses stereo/multi-channel by averaging channels.
    # Mono is required because WhisperProcessor expects a single-channel
    # waveform; stereo arrays would cause a shape mismatch in the feature
    # extractor.
    waveform, sr = librosa.load(path, sr=None, mono=True)

    # Resample only when the source rate differs from Whisper's expected rate.
    if sr != WHISPER_SAMPLE_RATE:
        waveform = librosa.resample(waveform, orig_sr=sr, target_sr=WHISPER_SAMPLE_RATE)

    return torch.from_numpy(waveform)


# ---------------------------------------------------------------------------
# Transcription
# ---------------------------------------------------------------------------


def transcribe(
    waveform: torch.Tensor,
    *,
    model_name: str,
    language: str | None,
    device: torch.device,
) -> str:
    """Run Whisper inference and return the decoded transcript string.

    Args:
        waveform:   1-D float32 tensor at 16 kHz.
        model_name: HuggingFace model ID, e.g. "openai/whisper-small".
        language:   BCP-47 language code ("en", "fr", …) or None for auto-detect.
        device:     torch device to run inference on.
    """
    # WhisperProcessor wraps two components:
    #   • feature_extractor: waveform → log-mel spectrogram tensor
    #   • tokenizer: token ids → human-readable text
    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(model_name).to(device)
    model.eval()  # disable dropout; we're only doing inference

    # Convert waveform to mel spectrogram. The processor expects a plain
    # Python list or numpy array, so we detach and convert.
    inputs = processor(
        waveform.numpy(),
        sampling_rate=WHISPER_SAMPLE_RATE,
        return_tensors="pt",  # return PyTorch tensors
    )
    # input_features shape: (1, 80, 3000) — batch × mel_bins × time_frames
    input_features = inputs.input_features.to(device)

    # forced_decoder_ids pins the language and task tokens at the start of
    # decoding. Setting task="transcribe" (vs "translate") keeps the output
    # in the source language. Pass language=None to let Whisper auto-detect.
    forced_decoder_ids = processor.get_decoder_prompt_ids(
        language=language,
        task="transcribe",
    )

    with torch.no_grad():
        # model.generate() runs beam search and returns token id sequences.
        predicted_ids = model.generate(
            input_features,
            forced_decoder_ids=forced_decoder_ids,
        )

    # Decode token ids back to a string; skip_special_tokens removes
    # language/task markers and padding tokens from the output.
    transcript = processor.batch_decode(predicted_ids, skip_special_tokens=True)
    return transcript[0].strip()


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Transcribe an audio file with Whisper.")
    parser.add_argument("--audio", required=True, help="Path to input audio file.")
    parser.add_argument(
        "--model",
        default="openai/whisper-small",
        help="HuggingFace model name (default: openai/whisper-small).",
    )
    parser.add_argument(
        "--language",
        default="en",
        help="Language code for transcription, e.g. 'en', 'fr'. Pass 'none' for auto-detect.",
    )
    args = parser.parse_args()

    device = _get_device()
    # Treat the string "none" (case-insensitive) as Python None for auto-detect.
    language = None if args.language.lower() == "none" else args.language

    waveform = load_audio(args.audio)
    transcript = transcribe(waveform, model_name=args.model, language=language, device=device)

    # Print transcript plainly first for easy piping / grep.
    print(transcript)

    # Then print full metadata as JSON, matching the style of src/inference.py.
    print(
        json.dumps(
            {
                "audio": args.audio,
                "model": args.model,
                "language": language,
                "device": str(device),
                "transcript": transcript,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

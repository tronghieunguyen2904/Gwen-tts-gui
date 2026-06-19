#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import soundfile as sf

from inference import (
    generate_voice_clone,
    generate_with_speaker,
    load_model,
    load_speaker_info,
)
from srt_utils import parse_srt_file


def _ensure_mono(wav: np.ndarray) -> np.ndarray:
    if wav.ndim == 1:
        return wav
    # If model returns (channels, n) or (n, channels), flatten to mono
    if wav.ndim == 2:
        if wav.shape[0] <= 2 and wav.shape[0] < wav.shape[1]:
            return wav.mean(axis=0)
        return wav.mean(axis=1)
    return wav.reshape(-1)


def _silence(sr: int, ms: int) -> np.ndarray:
    n = int(round(sr * (ms / 1000.0)))
    if n <= 0:
        return np.zeros((0,), dtype=np.float32)
    return np.zeros((n,), dtype=np.float32)


def synthesize_srt(
    *,
    model_path: str,
    device: str,
    language: str,
    srt_path: str,
    output_wav: str,
    speaker: Optional[str],
    ref_audio: Optional[str],
    ref_text: Optional[str],
    respect_timestamps: bool,
    gap_ms: int,
) -> Tuple[str, int, int]:
    entries = parse_srt_file(srt_path)
    if not entries:
        raise ValueError("SRT is empty or could not be parsed.")

    base_dir = Path(__file__).parent
    ref_info_path = base_dir / "data" / "ref_info.json"

    model = load_model(model_path, device=device)

    if speaker:
        ref_info = load_speaker_info(ref_info_path)
        synth_fn = lambda text: generate_with_speaker(  # noqa: E731
            model, text, language, speaker, ref_info, base_dir
        )
    else:
        if not ref_audio or not ref_text:
            raise ValueError("ref_audio and ref_text are required when speaker is not provided.")
        synth_fn = lambda text: generate_voice_clone(  # noqa: E731
            model, text, language, ref_audio, ref_text
        )

    out_chunks: list[np.ndarray] = []
    sr_out: Optional[int] = None
    cursor_ms = 0

    for e in entries:
        wav, sr = synth_fn(e.text)
        wav_np = np.asarray(wav, dtype=np.float32)
        wav_np = _ensure_mono(wav_np)

        if sr_out is None:
            sr_out = int(sr)
        elif int(sr) != sr_out:
            raise RuntimeError(f"Sample rate mismatch: got {sr}, expected {sr_out}")

        if respect_timestamps:
            if e.start_ms > cursor_ms:
                out_chunks.append(_silence(sr_out, e.start_ms - cursor_ms))
                cursor_ms = e.start_ms
            out_chunks.append(wav_np)
            cursor_ms += int(round((len(wav_np) / sr_out) * 1000.0))
        else:
            if out_chunks and gap_ms > 0:
                out_chunks.append(_silence(sr_out, gap_ms))
            out_chunks.append(wav_np)

    if sr_out is None:
        raise RuntimeError("No audio generated.")

    final = np.concatenate(out_chunks, axis=0) if out_chunks else np.zeros((0,), dtype=np.float32)
    sf.write(output_wav, final, sr_out)
    return output_wav, sr_out, len(entries)


def main():
    p = argparse.ArgumentParser(description="SRT to Speech (Gwen-TTS voice cloning)")
    p.add_argument("--srt", required=True, help="Path to .srt file")
    p.add_argument("--output", default="output_srt.wav", help="Output wav path")
    p.add_argument("--language", default="vietnamese", help="Language (default: vietnamese)")
    p.add_argument("--model_path", default="g-group-ai-lab/gwen-tts-0.6B", help="HF model id or path")
    p.add_argument("--device", default="cuda:0", help="Device (default: cuda:0)")

    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--speaker", default=None, help="Built-in speaker key (e.g., yen_nhi)")
    g.add_argument("--ref_audio", default=None, help="Reference wav for voice cloning")

    p.add_argument("--ref_text", default=None, help="Transcript for ref_audio (required if --ref_audio)")

    p.add_argument(
        "--respect_timestamps",
        action="store_true",
        help="Insert silence to match SRT start times",
    )
    p.add_argument(
        "--gap_ms",
        type=int,
        default=120,
        help="Gap (ms) between subtitle lines when not respecting timestamps (default: 120)",
    )

    args = p.parse_args()

    if args.ref_audio and not args.ref_text:
        p.error("--ref_text is required when using --ref_audio")

    out, sr, n = synthesize_srt(
        model_path=args.model_path,
        device=args.device,
        language=args.language,
        srt_path=args.srt,
        output_wav=args.output,
        speaker=args.speaker,
        ref_audio=args.ref_audio,
        ref_text=args.ref_text,
        respect_timestamps=args.respect_timestamps,
        gap_ms=args.gap_ms,
    )
    print(f"Saved: {out} | sr={sr} | lines={n}")


if __name__ == "__main__":
    main()
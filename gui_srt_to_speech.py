from __future__ import annotations

import json
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from srt_to_speech import synthesize_srt


def _load_ref_info() -> dict:
    base_dir = Path(__file__).parent
    p = base_dir / "data" / "ref_info.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


class App(ttk.Frame):
    def __init__(self, master: tk.Tk):
        super().__init__(master, padding=12)
        self.master = master
        self.grid(sticky="nsew")

        master.title("Gwen-TTS | SRT to Speech")
        master.geometry("880x640")
        master.columnconfigure(0, weight=1)
        master.rowconfigure(0, weight=1)

        self.ref_info = _load_ref_info()
        speaker_keys = sorted(self.ref_info.keys())

        self.model_path = tk.StringVar(value="g-group-ai-lab/gwen-tts-0.6B")
        self.device = tk.StringVar(value="cuda:0")
        self.language = tk.StringVar(value="vietnamese")

        self.mode = tk.StringVar(value="speaker")  # speaker | custom
        self.speaker = tk.StringVar(value=speaker_keys[0] if speaker_keys else "")

        self.ref_audio = tk.StringVar(value="")
        self.ref_text = tk.StringVar(value="")

        self.srt_path = tk.StringVar(value="")
        self.output_wav = tk.StringVar(value=str((Path(__file__).parent / "output_srt.wav").resolve()))
        self.respect_timestamps = tk.BooleanVar(value=True)
        self.gap_ms = tk.IntVar(value=120)

        self.status = tk.StringVar(value="Ready.")
        self._busy = False

        self._build_ui(speaker_keys)

    def _build_ui(self, speaker_keys: list[str]):
        self.columnconfigure(0, weight=1)

        lf_model = ttk.LabelFrame(self, text="Model")
        lf_model.grid(row=0, column=0, sticky="ew", padx=4, pady=6)
        lf_model.columnconfigure(1, weight=1)

        ttk.Label(lf_model, text="Model path / HF id").grid(row=0, column=0, sticky="w")
        ttk.Entry(lf_model, textvariable=self.model_path).grid(row=0, column=1, sticky="ew", padx=6)

        ttk.Label(lf_model, text="Device").grid(row=1, column=0, sticky="w")
        ttk.Entry(lf_model, textvariable=self.device).grid(row=1, column=1, sticky="ew", padx=6)

        ttk.Label(lf_model, text="Language").grid(row=2, column=0, sticky="w")
        ttk.Entry(lf_model, textvariable=self.language).grid(row=2, column=1, sticky="ew", padx=6)

        lf_voice = ttk.LabelFrame(self, text="Voice")
        lf_voice.grid(row=1, column=0, sticky="ew", padx=4, pady=6)
        lf_voice.columnconfigure(1, weight=1)

        rb1 = ttk.Radiobutton(lf_voice, text="Built-in speaker", value="speaker", variable=self.mode, command=self._refresh_mode)
        rb2 = ttk.Radiobutton(lf_voice, text="Custom voice (ref_audio + ref_text)", value="custom", variable=self.mode, command=self._refresh_mode)
        rb1.grid(row=0, column=0, sticky="w", columnspan=2)
        rb2.grid(row=1, column=0, sticky="w", columnspan=2)

        ttk.Label(lf_voice, text="Speaker").grid(row=2, column=0, sticky="w")
        self.speaker_combo = ttk.Combobox(lf_voice, textvariable=self.speaker, values=speaker_keys, state="readonly" if speaker_keys else "disabled")
        self.speaker_combo.grid(row=2, column=1, sticky="ew", padx=6)

        ttk.Label(lf_voice, text="Ref audio (.wav)").grid(row=3, column=0, sticky="w")
        self.ref_audio_entry = ttk.Entry(lf_voice, textvariable=self.ref_audio)
        self.ref_audio_entry.grid(row=3, column=1, sticky="ew", padx=6)
        ttk.Button(lf_voice, text="Browse", command=self._browse_ref_audio).grid(row=3, column=2, padx=4)

        ttk.Label(lf_voice, text="Ref text (transcript)").grid(row=4, column=0, sticky="nw")
        self.ref_text_box = tk.Text(lf_voice, height=5, wrap="word")
        self.ref_text_box.grid(row=4, column=1, columnspan=2, sticky="ew", padx=6, pady=4)

        lf_io = ttk.LabelFrame(self, text="SRT → Speech")
        lf_io.grid(row=2, column=0, sticky="nsew", padx=4, pady=6)
        lf_io.columnconfigure(1, weight=1)
        self.rowconfigure(2, weight=1)

        ttk.Label(lf_io, text="SRT file").grid(row=0, column=0, sticky="w")
        ttk.Entry(lf_io, textvariable=self.srt_path).grid(row=0, column=1, sticky="ew", padx=6)
        ttk.Button(lf_io, text="Browse", command=self._browse_srt).grid(row=0, column=2, padx=4)

        ttk.Label(lf_io, text="Output wav").grid(row=1, column=0, sticky="w")
        ttk.Entry(lf_io, textvariable=self.output_wav).grid(row=1, column=1, sticky="ew", padx=6)
        ttk.Button(lf_io, text="Browse", command=self._browse_output).grid(row=1, column=2, padx=4)

        ttk.Checkbutton(lf_io, text="Respect SRT timestamps (insert silence)", variable=self.respect_timestamps).grid(
            row=2, column=0, columnspan=2, sticky="w", pady=(6, 0)
        )
        ttk.Label(lf_io, text="Gap ms (if not respecting timestamps)").grid(row=3, column=0, sticky="w")
        ttk.Spinbox(lf_io, from_=0, to=2000, textvariable=self.gap_ms, width=10).grid(row=3, column=1, sticky="w", padx=6)

        btns = ttk.Frame(lf_io)
        btns.grid(row=4, column=0, columnspan=3, sticky="ew", pady=10)
        btns.columnconfigure(0, weight=1)
        self.run_btn = ttk.Button(btns, text="Generate", command=self._run)
        self.run_btn.grid(row=0, column=1, sticky="e")

        status_bar = ttk.Frame(self)
        status_bar.grid(row=3, column=0, sticky="ew", padx=4, pady=(6, 0))
        status_bar.columnconfigure(0, weight=1)
        ttk.Label(status_bar, textvariable=self.status).grid(row=0, column=0, sticky="w")

        self._refresh_mode()

    def _refresh_mode(self):
        is_speaker = self.mode.get() == "speaker"
        self.speaker_combo.configure(state="readonly" if is_speaker and self.speaker_combo["values"] else "disabled")

        state_custom = "disabled" if is_speaker else "normal"
        self.ref_audio_entry.configure(state=state_custom)
        if is_speaker:
            self.ref_text_box.configure(state="disabled")
        else:
            self.ref_text_box.configure(state="normal")

    def _browse_ref_audio(self):
        p = filedialog.askopenfilename(title="Select reference wav", filetypes=[("WAV", "*.wav"), ("All files", "*.*")])
        if p:
            self.ref_audio.set(p)

    def _browse_srt(self):
        p = filedialog.askopenfilename(title="Select SRT", filetypes=[("SRT", "*.srt"), ("All files", "*.*")])
        if p:
            self.srt_path.set(p)

    def _browse_output(self):
        p = filedialog.asksaveasfilename(
            title="Save output wav",
            defaultextension=".wav",
            filetypes=[("WAV", "*.wav")],
            initialfile="output_srt.wav",
        )
        if p:
            self.output_wav.set(p)

    def _set_busy(self, busy: bool):
        self._busy = busy
        self.run_btn.configure(state="disabled" if busy else "normal")

    def _run(self):
        if self._busy:
            return

        srt_path = self.srt_path.get().strip()
        out = self.output_wav.get().strip()
        if not srt_path:
            messagebox.showerror("Missing input", "Please select an .srt file.")
            return
        if not out:
            messagebox.showerror("Missing output", "Please choose an output .wav path.")
            return

        speaker = self.speaker.get().strip() if self.mode.get() == "speaker" else None
        ref_audio = self.ref_audio.get().strip() if self.mode.get() == "custom" else None
        ref_text = None
        if self.mode.get() == "custom":
            ref_text = self.ref_text_box.get("1.0", "end").strip()
            if not ref_audio:
                messagebox.showerror("Missing ref audio", "Please select reference audio (.wav).")
                return
            if not ref_text:
                messagebox.showerror("Missing ref text", "Please paste transcript for the reference audio (ref_text).")
                return

        self._set_busy(True)
        self.status.set("Loading model & generating... (this may take a while)")

        def work():
            try:
                out_path, sr, n = synthesize_srt(
                    model_path=self.model_path.get().strip(),
                    device=self.device.get().strip(),
                    language=self.language.get().strip(),
                    srt_path=srt_path,
                    output_wav=out,
                    speaker=speaker,
                    ref_audio=ref_audio,
                    ref_text=ref_text,
                    respect_timestamps=bool(self.respect_timestamps.get()),
                    gap_ms=int(self.gap_ms.get()),
                )
                self.master.after(0, lambda: self._done_ok(out_path, sr, n))
            except Exception as e:
                # Python clears `e` at end of `except`; capture for Tk callback.
                self.master.after(0, lambda err=e: self._done_err(err))

        threading.Thread(target=work, daemon=True).start()

    def _done_ok(self, out_path: str, sr: int, n: int):
        self._set_busy(False)
        self.status.set(f"Done. Saved: {out_path} | sr={sr} | lines={n}")
        messagebox.showinfo("Done", f"Saved:\n{out_path}\n\nSample rate: {sr}\nLines: {n}")

    def _done_err(self, e: Exception):
        self._set_busy(False)
        self.status.set("Error.")
        messagebox.showerror("Error", str(e))


def main():
    root = tk.Tk()
    style = ttk.Style(root)
    if "vista" in style.theme_names():
        style.theme_use("vista")
    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
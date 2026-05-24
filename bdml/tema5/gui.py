from __future__ import annotations

import threading
import tkinter as tk
from pathlib import Path
from queue import Empty, Queue
from tkinter import filedialog, messagebox, ttk

import torch
import torchaudio

try:
    import sounddevice as sd
except ImportError:
    sd = None

try:
    from .constants import DEFAULT_CHECKPOINT_PATH, SAMPLE_RATE
    from .inference import (
        build_model_from_checkpoint,
        transcribe_file,
        transcribe_waveform,
    )
except ImportError:
    from constants import DEFAULT_CHECKPOINT_PATH, SAMPLE_RATE
    from inference import (
        build_model_from_checkpoint,
        transcribe_file,
        transcribe_waveform,
    )


class SpeechToTextApp:
    def __init__(self, checkpoint_path: str | Path = DEFAULT_CHECKPOINT_PATH):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_path = Path(checkpoint_path)
        self.model = None
        self.tokenizer = None
        self.feature_extractor = None

        self.recorded_waveform: torch.Tensor | None = None
        self.recorded_sample_rate = SAMPLE_RATE
        self.selected_file: Path | None = None

        self.queue: Queue[tuple[str, str]] = Queue()
        self.root = tk.Tk()
        self.root.title("LibriSpeech LSTM Speech-to-Text")
        self.root.geometry("920x650")
        self.root.configure(bg="#0f172a")

        self._configure_styles()
        self._build_layout()
        self._try_load_default_checkpoint()

    def _configure_styles(self) -> None:
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("Root.TFrame", background="#0f172a")
        style.configure("Panel.TFrame", background="#111827")
        style.configure(
            "Title.TLabel",
            background="#0f172a",
            foreground="#f8fafc",
            font=("Segoe UI", 20, "bold"),
        )
        style.configure(
            "Body.TLabel",
            background="#111827",
            foreground="#e5e7eb",
            font=("Segoe UI", 11),
        )
        style.configure(
            "Info.TLabel",
            background="#111827",
            foreground="#93c5fd",
            font=("Segoe UI", 10),
        )
        style.configure("Accent.TButton", font=("Segoe UI", 10, "bold"), padding=10)

    def _build_layout(self) -> None:
        root_panel = ttk.Frame(self.root, style="Root.TFrame", padding=20)
        root_panel.pack(fill="both", expand=True)

        ttk.Label(
            root_panel, text="Speech-to-Text (LSTM + CTC)", style="Title.TLabel"
        ).pack(anchor="w", pady=(0, 16))

        body = ttk.Frame(root_panel, style="Root.TFrame")
        body.pack(fill="both", expand=True)
        body.columnconfigure(0, weight=1)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)

        left_panel = ttk.Frame(body, style="Panel.TFrame", padding=16)
        left_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        right_panel = ttk.Frame(body, style="Panel.TFrame", padding=16)
        right_panel.grid(row=0, column=1, sticky="nsew", padx=(10, 0))

        self.status_label = ttk.Label(
            left_panel,
            text="Ready",
            style="Info.TLabel",
            justify="left",
        )
        self.status_label.pack(fill="x", pady=(0, 12))

        self.duration_var = tk.StringVar(value="5")
        duration_row = ttk.Frame(left_panel, style="Panel.TFrame")
        duration_row.pack(fill="x", pady=(0, 12))
        ttk.Label(duration_row, text="Record Seconds:", style="Body.TLabel").pack(
            side="left"
        )
        ttk.Entry(duration_row, textvariable=self.duration_var, width=8).pack(
            side="left", padx=(8, 0)
        )

        ttk.Button(
            left_panel,
            text="Record Audio",
            style="Accent.TButton",
            command=self.record_audio,
        ).pack(fill="x", pady=(0, 8))

        ttk.Button(
            left_panel,
            text="Transcribe Recording",
            style="Accent.TButton",
            command=self.transcribe_recording,
        ).pack(fill="x", pady=(0, 8))

        ttk.Button(
            left_panel,
            text="Load Audio File",
            style="Accent.TButton",
            command=self.load_audio_file,
        ).pack(fill="x", pady=(0, 8))

        ttk.Button(
            left_panel,
            text="Transcribe File",
            style="Accent.TButton",
            command=self.transcribe_loaded_file,
        ).pack(fill="x", pady=(0, 8))

        ttk.Button(
            left_panel,
            text="Load Checkpoint",
            style="Accent.TButton",
            command=self.load_checkpoint_from_dialog,
        ).pack(fill="x", pady=(0, 8))

        ttk.Label(right_panel, text="Transcription", style="Body.TLabel").pack(
            anchor="w"
        )
        self.output = tk.Text(
            right_panel,
            wrap="word",
            height=20,
            bg="#0b1220",
            fg="#e5e7eb",
            insertbackground="#e5e7eb",
            relief="flat",
            padx=12,
            pady=12,
        )
        self.output.pack(fill="both", expand=True, pady=(8, 0))
        self._set_output("Load a checkpoint and record audio to transcribe.")

    def _set_output(self, text: str) -> None:
        self.output.configure(state="normal")
        self.output.delete("1.0", tk.END)
        self.output.insert(tk.END, text)
        self.output.configure(state="disabled")

    def _set_status(self, text: str) -> None:
        self.status_label.configure(text=text)

    def _try_load_default_checkpoint(self) -> None:
        if not self.checkpoint_path.exists():
            self._set_status(f"No checkpoint found at {self.checkpoint_path}")
            return
        try:
            self._load_checkpoint(self.checkpoint_path)
            self._set_status(f"Loaded checkpoint: {self.checkpoint_path}")
        except Exception as exc:
            self._set_status(f"Failed to load checkpoint: {exc}")

    def _load_checkpoint(self, checkpoint_path: Path) -> None:
        self.model, self.tokenizer, self.feature_extractor, _ = (
            build_model_from_checkpoint(
                checkpoint_path,
                self.device,
            )
        )
        self.checkpoint_path = checkpoint_path

    def load_checkpoint_from_dialog(self) -> None:
        file_path = filedialog.askopenfilename(
            title="Select checkpoint",
            filetypes=[("PyTorch checkpoint", "*.pt *.pth"), ("All files", "*.*")],
        )
        if not file_path:
            return
        try:
            self._load_checkpoint(Path(file_path))
            self._set_status(f"Loaded checkpoint: {file_path}")
        except Exception as exc:
            messagebox.showerror("Checkpoint error", str(exc))

    def record_audio(self) -> None:
        if sd is None:
            messagebox.showerror(
                "Missing package",
                "Recording requires the sounddevice package.",
            )
            return

        try:
            seconds = float(self.duration_var.get())
            if seconds <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror(
                "Invalid duration", "Duration must be a positive number"
            )
            return

        def worker() -> None:
            try:
                self.queue.put(("status", "Recording in progress..."))
                self.root.after(0, self._poll_queue)
                frames = int(seconds * SAMPLE_RATE)
                recording = sd.rec(
                    frames, samplerate=SAMPLE_RATE, channels=1, dtype="float32"
                )
                sd.wait()
                waveform = torch.from_numpy(recording.T)
                self.recorded_waveform = waveform
                self.recorded_sample_rate = SAMPLE_RATE
                self.queue.put(("status", f"Recording finished ({seconds:.1f}s)"))
                self.root.after(0, self._poll_queue)
            except Exception as exc:
                self.queue.put(("error", str(exc)))
                self.root.after(0, self._poll_queue)

        threading.Thread(target=worker, daemon=True).start()

    def _poll_queue(self) -> None:
        while True:
            try:
                event, payload = self.queue.get_nowait()
            except Empty:
                break

            if event == "status":
                self._set_status(payload)
            elif event == "error":
                messagebox.showerror("Recording error", payload)

    def load_audio_file(self) -> None:
        file_path = filedialog.askopenfilename(
            title="Select audio file",
            filetypes=[("Audio files", "*.wav *.flac *.mp3"), ("All files", "*.*")],
        )
        if not file_path:
            return
        self.selected_file = Path(file_path)
        self._set_status(f"Loaded file: {self.selected_file}")

    def _ensure_model_ready(self) -> bool:
        if (
            self.model is None
            or self.feature_extractor is None
            or self.tokenizer is None
        ):
            messagebox.showerror(
                "Model not loaded",
                "Load a trained checkpoint before transcribing.",
            )
            return False
        return True

    def transcribe_recording(self) -> None:
        if not self._ensure_model_ready():
            return
        if self.recorded_waveform is None:
            messagebox.showinfo("No recording", "Record audio first.")
            return

        try:
            text = transcribe_waveform(
                model=self.model,
                waveform=self.recorded_waveform,
                sample_rate=self.recorded_sample_rate,
                feature_extractor=self.feature_extractor,
                tokenizer=self.tokenizer,
                device=self.device,
            )
            self._set_output(text if text else "(empty transcription)")
            self._set_status("Recording transcribed")
        except Exception as exc:
            messagebox.showerror("Transcription error", str(exc))

    def transcribe_loaded_file(self) -> None:
        if not self._ensure_model_ready():
            return
        if self.selected_file is None:
            messagebox.showinfo("No file", "Load an audio file first.")
            return

        try:
            text = transcribe_file(
                model=self.model,
                audio_path=self.selected_file,
                feature_extractor=self.feature_extractor,
                tokenizer=self.tokenizer,
                device=self.device,
            )
            self._set_output(text if text else "(empty transcription)")
            self._set_status(f"Transcribed: {self.selected_file}")
        except Exception as exc:
            messagebox.showerror("Transcription error", str(exc))

    def run(self) -> None:
        self.root.mainloop()

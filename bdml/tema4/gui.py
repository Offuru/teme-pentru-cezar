from __future__ import annotations

import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from queue import Queue, Empty

from PIL import Image, ImageTk
import torch

try:
    from .constants import DEFAULT_CHECKPOINT_PATH, DEFAULT_DATA_DIR
    from .data import get_dataloaders, train_model
    from .inference import predict_top_k
    from .model import create_model, load_checkpoint, save_checkpoint
except ImportError:
    from constants import DEFAULT_CHECKPOINT_PATH, DEFAULT_DATA_DIR
    from data import get_dataloaders, train_model
    from inference import predict_top_k
    from model import create_model, load_checkpoint, save_checkpoint


class CIFAR10ClassifierApp:
    def __init__(self, checkpoint_path: str | Path = DEFAULT_CHECKPOINT_PATH):
        self.checkpoint_path = Path(checkpoint_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = create_model(self.device)
        self.selected_image_path: Path | None = None
        self.preview_image = None
        self.training_queue: Queue[dict[str, float | int | str]] = Queue()
        self.training_running = False

        self.root = tk.Tk()
        self.root.title("CIFAR-10 CNN Classifier")
        self.root.geometry("900x640")
        self.root.minsize(820, 580)
        self.root.configure(bg="#0f172a")

        self._configure_style()
        self._build_layout()
        self._load_default_checkpoint()

    def _configure_style(self) -> None:
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
        container = ttk.Frame(self.root, style="Root.TFrame", padding=20)
        container.pack(fill="both", expand=True)

        header = ttk.Label(
            container,
            text="CIFAR-10 CNN Classifier",
            style="Title.TLabel",
        )
        header.pack(anchor="w", pady=(0, 16))

        body = ttk.Frame(container, style="Root.TFrame")
        body.pack(fill="both", expand=True)
        body.columnconfigure(0, weight=1)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)

        left_panel = ttk.Frame(body, style="Panel.TFrame", padding=16)
        left_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        right_panel = ttk.Frame(body, style="Panel.TFrame", padding=16)
        right_panel.grid(row=0, column=1, sticky="nsew", padx=(10, 0))

        self.preview_label = ttk.Label(
            left_panel, text="No image loaded", style="Body.TLabel", anchor="center"
        )
        self.preview_label.pack(fill="both", expand=True)

        self.image_path_label = ttk.Label(
            left_panel,
            text="Load a local image to classify it.",
            style="Info.TLabel",
            wraplength=360,
            justify="left",
        )
        self.image_path_label.pack(fill="x", pady=(12, 0))

        button_row = ttk.Frame(left_panel, style="Panel.TFrame")
        button_row.pack(fill="x", pady=(16, 0))

        ttk.Button(
            button_row,
            text="Load Image",
            style="Accent.TButton",
            command=self.load_image,
        ).pack(fill="x", pady=(0, 8))
        ttk.Button(
            button_row,
            text="Classify",
            style="Accent.TButton",
            command=self.classify_loaded_image,
        ).pack(fill="x", pady=(0, 8))
        ttk.Button(
            button_row,
            text="Load Checkpoint",
            style="Accent.TButton",
            command=self.choose_checkpoint,
        ).pack(fill="x", pady=(0, 8))
        ttk.Button(
            button_row,
            text="Train Model",
            style="Accent.TButton",
            command=self.train_model_in_background,
        ).pack(fill="x")

        model_info = (
            f"Device: {self.device}\n"
            f"Checkpoint: {self.checkpoint_path}\n"
            f"Data directory: {DEFAULT_DATA_DIR}"
        )
        ttk.Label(right_panel, text="Model Status", style="Body.TLabel").pack(
            anchor="w"
        )
        self.status_label = ttk.Label(
            right_panel, text=model_info, style="Info.TLabel", justify="left"
        )
        self.status_label.pack(fill="x", pady=(8, 16))

        ttk.Label(right_panel, text="Top-5 Predictions", style="Body.TLabel").pack(
            anchor="w"
        )
        self.progress_var = tk.DoubleVar(value=0.0)
        self.progress_label_var = tk.StringVar(value="Training idle")
        self.progress_bar = ttk.Progressbar(
            right_panel,
            orient="horizontal",
            mode="determinate",
            maximum=100,
            variable=self.progress_var,
        )
        self.progress_bar.pack(fill="x", pady=(8, 8))
        self.progress_label = ttk.Label(
            right_panel,
            textvariable=self.progress_label_var,
            style="Info.TLabel",
            justify="left",
        )
        self.progress_label.pack(fill="x", pady=(0, 8))

        self.predictions_text = tk.Text(
            right_panel,
            height=18,
            wrap="word",
            bg="#0b1220",
            fg="#e5e7eb",
            insertbackground="#e5e7eb",
            relief="flat",
            padx=12,
            pady=12,
        )
        self.predictions_text.pack(fill="both", expand=True, pady=(8, 0))
        self._write_predictions([("No predictions yet", 0.0)])

    def _load_default_checkpoint(self) -> None:
        if self.checkpoint_path.exists():
            try:
                load_checkpoint(
                    self.model, self.checkpoint_path, map_location=self.device
                )
                self.model.to(self.device)
                self._set_status(f"Loaded checkpoint: {self.checkpoint_path}")
            except Exception as exc:
                self._set_status(f"Failed to load checkpoint: {exc}")
        else:
            self._set_status(
                "No checkpoint found. Use 'Train Model' or 'Load Checkpoint' before classifying."
            )

    def _set_status(self, message: str) -> None:
        self.status_label.configure(
            text=f"Device: {self.device}\nCheckpoint: {self.checkpoint_path}\n{message}"
        )

    def _write_predictions(self, predictions: list[tuple[str, float]]) -> None:
        self.predictions_text.configure(state="normal")
        self.predictions_text.delete("1.0", tk.END)
        for index, (label, score) in enumerate(predictions, start=1):
            self.predictions_text.insert(
                tk.END, f"{index}. {label:<12} {score * 100:6.2f}%\n"
            )
        self.predictions_text.configure(state="disabled")

    def load_image(self) -> None:
        file_path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp *.webp"),
                ("All files", "*.*"),
            ],
        )
        if not file_path:
            return

        self.selected_image_path = Path(file_path)
        self._show_preview(self.selected_image_path)
        self.image_path_label.configure(text=str(self.selected_image_path))
        self._set_status(f"Loaded image: {self.selected_image_path}")

    def _show_preview(self, image_path: Path) -> None:
        image = Image.open(image_path).convert("RGB")
        image.thumbnail((360, 360))
        self.preview_image = ImageTk.PhotoImage(image)
        self.preview_label.configure(image=self.preview_image, text="")

    def classify_loaded_image(self) -> None:
        if self.selected_image_path is None:
            messagebox.showinfo("No image", "Load an image before classifying it.")
            return

        try:
            predictions = predict_top_k(
                self.model, self.selected_image_path, self.device, top_k=5
            )
        except Exception as exc:
            messagebox.showerror("Classification error", str(exc))
            return

        self._write_predictions(predictions)
        self._set_status(f"Classified image: {self.selected_image_path}")

    def choose_checkpoint(self) -> None:
        file_path = filedialog.askopenfilename(
            title="Select model checkpoint",
            filetypes=[("PyTorch checkpoint", "*.pt *.pth"), ("All files", "*.*")],
        )
        if not file_path:
            return

        checkpoint_path = Path(file_path)
        try:
            load_checkpoint(self.model, checkpoint_path, map_location=self.device)
            self.model.to(self.device)
            self.checkpoint_path = checkpoint_path
            self._set_status(f"Loaded checkpoint: {self.checkpoint_path}")
        except Exception as exc:
            messagebox.showerror("Checkpoint error", str(exc))

    def train_model_in_background(self) -> None:
        if self.training_running:
            messagebox.showinfo("Training", "Training is already running.")
            return

        self.training_running = True
        self.progress_var.set(0)
        self.progress_label_var.set("Preparing training run...")

        def progress_callback(payload: dict[str, float | int | str]) -> None:
            self.training_queue.put(payload)
            self.root.after(0, self._drain_training_queue)

        def worker() -> None:
            try:
                self.root.after(0, lambda: self._set_status("Training model..."))
                train_loader, validation_loader = get_dataloaders(
                    DEFAULT_DATA_DIR, batch_size=128, download=True
                )
                history = train_model(
                    self.model,
                    train_loader,
                    validation_loader,
                    self.device,
                    epochs=15,
                    progress_callback=progress_callback,
                )
                save_checkpoint(
                    self.model, self.checkpoint_path, {"device": str(self.device)}
                )
                self.root.after(
                    0,
                    lambda: self._set_status(
                        f"Training finished and saved to {self.checkpoint_path}"
                    ),
                )
                self.root.after(
                    0,
                    lambda: self.progress_label_var.set(
                        f"Finished. Final validation accuracy: {history.validation_accuracy[-1]:.3f}"
                    ),
                )
            except Exception as exc:
                self.root.after(
                    0, lambda: messagebox.showerror("Training error", str(exc))
                )
            finally:
                self.root.after(0, self._mark_training_finished)

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()

    def _drain_training_queue(self) -> None:
        latest_payload: dict[str, float | int | str] | None = None
        while True:
            try:
                latest_payload = self.training_queue.get_nowait()
            except Empty:
                break

        if latest_payload is None:
            return

        phase = str(latest_payload.get("phase", ""))
        if phase == "train_batch":
            epoch = int(latest_payload["epoch"])
            epochs = int(latest_payload["epochs"])
            batch = int(latest_payload["batch"])
            batches = int(latest_payload["batches"])
            batch_loss = float(latest_payload["batch_loss"])
            batch_accuracy = float(latest_payload["batch_accuracy"])
            overall = ((epoch - 1) + batch / batches) / max(1, epochs) * 100.0
            self.progress_var.set(overall)
            self.progress_label_var.set(
                f"Epoch {epoch}/{epochs} | Batch {batch}/{batches} | loss {batch_loss:.4f} | acc {batch_accuracy:.3f}"
            )
        elif phase == "epoch_end":
            epoch = int(latest_payload["epoch"])
            epochs = int(latest_payload["epochs"])
            train_loss = float(latest_payload["train_loss"])
            train_accuracy = float(latest_payload["train_accuracy"])
            validation_loss = float(latest_payload["validation_loss"])
            validation_accuracy = float(latest_payload["validation_accuracy"])
            self.progress_var.set(epoch / max(1, epochs) * 100.0)
            self.progress_label_var.set(
                f"Epoch {epoch}/{epochs} done | train loss {train_loss:.4f} | train acc {train_accuracy:.3f} | val loss {validation_loss:.4f} | val acc {validation_accuracy:.3f}"
            )

    def _mark_training_finished(self) -> None:
        self.training_running = False
        if self.progress_var.get() < 100:
            self.progress_var.set(100)

    def run(self) -> None:
        self.root.mainloop()

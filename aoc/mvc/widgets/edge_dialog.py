import tkinter as tk
from tkinter import messagebox


class EdgeDialogWidget:
    """Custom dialog for inputting edge capacity and flow."""

    def __init__(self, parent, start: int, end: int):
        self.result = None

        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Add Edge")
        self.dialog.transient(parent)
        self.dialog.grab_set()
        self.width = 300
        self.height = 200

        self.dialog.geometry(f"{self.width}x{self.height}")

        label = tk.Label(
            self.dialog, text=f"Add edge {start} → {end}", font=("Arial", 11, "bold")
        )
        label.pack(pady=10)

        input_frame = tk.Frame(self.dialog)
        input_frame.pack(pady=10)

        tk.Label(input_frame, text="Flow:", width=10, anchor="w").grid(
            row=0, column=0, padx=5, pady=5
        )
        self.flow_entry = tk.Entry(input_frame, width=15)
        self.flow_entry.insert(0, "0")
        self.flow_entry.grid(row=0, column=1, padx=5, pady=5)

        tk.Label(input_frame, text="Capacity:", width=10, anchor="w").grid(
            row=1, column=0, padx=5, pady=5
        )
        self.capacity_entry = tk.Entry(input_frame, width=15)
        self.capacity_entry.insert(0, "10")
        self.capacity_entry.grid(row=1, column=1, padx=5, pady=5)

        button_frame = tk.Frame(self.dialog)
        button_frame.pack(pady=10)

        ok_button = tk.Button(button_frame, text="OK", command=self._on_ok, width=10)
        ok_button.pack(side=tk.LEFT, padx=5)

        cancel_button = tk.Button(
            button_frame, text="Cancel", command=self._on_cancel, width=10
        )
        cancel_button.pack(side=tk.LEFT, padx=5)

        self.dialog.bind("<Return>", lambda e: self._on_ok())
        self.dialog.bind("<Escape>", lambda e: self._on_cancel())

        self.flow_entry.focus_set()

        parent.wait_window(self.dialog)

    def _validate_input(self) -> bool:
        """Validate that inputs are valid non-negative integers and capacity >= flow."""
        try:
            capacity = int(self.capacity_entry.get())
            flow = int(self.flow_entry.get())

            if capacity < 0:
                messagebox.showerror(
                    "Invalid Input",
                    "Capacity must be non-negative.",
                    parent=self.dialog,
                )
                return False

            if flow < 0:
                messagebox.showerror(
                    "Invalid Input", "Flow must be non-negative.", parent=self.dialog
                )
                return False

            if flow > capacity:
                messagebox.showerror(
                    "Invalid Input", "Flow cannot exceed capacity.", parent=self.dialog
                )
                return False

            return True

        except ValueError:
            messagebox.showerror(
                "Invalid Input",
                "Please enter valid integers for capacity and flow.",
                parent=self.dialog,
            )
            return False

    def _on_ok(self):
        """Handle OK button click."""
        if self._validate_input():
            capacity = int(self.capacity_entry.get())
            flow = int(self.flow_entry.get())
            self.result = (capacity, flow)
            self.dialog.destroy()

    def _on_cancel(self):
        """Handle Cancel button click."""
        self.result = None
        self.dialog.destroy()

    def get_result(self):
        """Return the result tuple (capacity, flow) or None if cancelled."""
        return self.result

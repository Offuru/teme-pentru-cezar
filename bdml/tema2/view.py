from controller import Controller
import tkinter as tk
import numpy as np


class View:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("K Nearest Neighbours")
        self.width = self.root.winfo_screenwidth() - 100
        self.height = self.root.winfo_screenheight() - 100
        self.root.state("zoomed")
        self.k = 3
        self.weighted = False

        n_points = 100
        self.controller = Controller(self.width, self.height, n_points)

        self.point_radius = 4

        self.input_frame = None

        self.weighted_label = tk.Label(
            self.root, text="Weighted: False", font=("Arial", 12), anchor="w"
        )
        self.weighted_label.pack(side=tk.TOP, anchor="w", padx=10, pady=5)

        self.k_label = tk.Label(
            self.root, text="K: 3", font=("Arial", 12), anchor="w", padx=10, pady=5
        )
        self.k_label.pack(side=tk.TOP, anchor="w", padx=10, pady=5)

        self.canvas = tk.Canvas(
            self.root,
            width=self.width,
            height=self.height,
            bg="white",
        )

        self.canvas.pack(
            side=tk.BOTTOM,
            fill=tk.BOTH,
            expand=True,
            padx=10,
            pady=10,
        )

        self.root.bind("<k>", self.create_k_input)
        self.root.bind("<K>", self.create_k_input)
        self.root.bind("<h>", self.create_validation_window)
        self.root.bind("<H>", self.create_validation_window)
        self.canvas.bind("<Button-1>", self._on_left_click)
        self.root.bind("<space>", self._on_spacebar_click)

        self.render()

    def _on_left_click(self, event: tk.Event):
        self.controller.add_point((event.x, event.y), self.k, self.weighted)
        self.render()

    def _on_spacebar_click(self, event: tk.Event):
        self.weighted = not self.weighted
        self.weighted_label.config(text=f"Weighted: {self.weighted}")

    def create_k_input(self, event: tk.Event):

        if self.input_frame is not None:
            return
        self.input_frame = tk.Toplevel(self.root)
        self.input_frame.geometry("150x50")

        label = tk.Label(self.input_frame, text="K: ")
        label.pack(side=tk.LEFT, padx=5)

        self.k_entry = tk.Entry(self.input_frame, width=10)
        self.k_entry.pack(side=tk.LEFT, padx=5)
        self.k_entry.insert(0, str(self.k))
        self.k_entry.focus_set()

        button = tk.Button(self.input_frame, text="Set K", command=self._set_k)
        button.pack(side=tk.LEFT, padx=5)

    def create_validation_window(self, event: tk.Event):
        validation_window = tk.Toplevel(self.root)
        validation_window.geometry("200x100")

        label = tk.Label(validation_window, text="Max K: ")
        label.pack(side=tk.LEFT, padx=5)

        max_k_entry = tk.Entry(validation_window, width=10)
        max_k_entry.pack(side=tk.LEFT, padx=5)
        max_k_entry.insert(0, "10")
        max_k_entry.focus_set()

        def run_cross_validation():
            try:
                max_k = int(max_k_entry.get())
                best_k, best_acc = self.controller.cross_validation(
                    max_k, self.weighted
                )
                result_label.config(text=f"Best K: {best_k}, Accuracy: {best_acc:.2f}")
            except Exception as e:
                result_label.config(text=f"Error: {str(e)}")

        button = tk.Button(validation_window, text="Run", command=run_cross_validation)
        button.pack(side=tk.LEFT, padx=5)

        result_label = tk.Label(validation_window, text="")
        result_label.pack(side=tk.BOTTOM, pady=10)

    def _set_k(self):
        try:
            initial_value = self.k
            self.k = int(self.k_entry.get())
        except Exception:
            self.k = initial_value
        finally:
            self.k_label.config(text=f"K: {self.k}")
            if self.input_frame:
                self.input_frame.destroy()
                self.input_frame = None

    def render(self):
        self.canvas.delete("all")

        for idx, point in enumerate(self.controller.points):
            start = point - self.point_radius
            end = point + self.point_radius
            self.canvas.create_oval(
                *start,
                *end,
                fill="green" if self.controller.labels[idx] == 1 else "red",
                outline="black",
            )

    def run(self):
        self.root.mainloop()

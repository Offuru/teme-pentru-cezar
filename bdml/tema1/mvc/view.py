from .controller import Controller
import tkinter as tk
import numpy as np


class View:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("K-Means")
        self.width = self.root.winfo_screenwidth() - 100
        self.height = self.root.winfo_screenheight() - 100
        self.root.state("zoomed")

        n_points = 100
        max_iter = 100
        self.controller = Controller(self.width, self.height, n_points, max_iter)

        self.point_radius = 3
        self.centroid_radius = 10
        self.min_centroid_distance = 5 + self.centroid_radius
        self.bounding_box_width = 2
        self.line_width = 2

        self.canvas = tk.Canvas(
            self.root, width=self.width, height=self.height, bg="white"
        )
        self.canvas.pack(padx=10, pady=10)

        self.canvas.bind("<Button-1>", self._on_left_click)
        self.canvas.bind("<Button-3>", self._on_right_click)
        self.root.bind("<space>", self._on_spacebar_click)

        self.render()

    def _on_left_click(self, event: tk.Event):
        for centroid in self.controller.centroids[:]:
            if (
                np.linalg.norm(centroid - np.array([event.x, event.y]))
                < self.min_centroid_distance
            ):
                return

        self.controller.add_centroid((event.x, event.y))
        self.render()

    def _on_right_click(self, event: tk.Event):
        for idx, centroid in enumerate(self.controller.centroids[:]):
            if (
                np.linalg.norm(centroid - np.array([event.x, event.y]))
                < self.min_centroid_distance
            ):
                self.controller.remove_centroid(idx)
        self.render()

    def _on_spacebar_click(self, event: tk.Event):
        self.controller.run_kmeans()
        self.render()

    def render(self):
        self.canvas.delete("all")
        for point in self.controller.points:
            start = point - self.point_radius
            end = point + self.point_radius
            self.canvas.create_oval(*start, *end, fill="gray", outline="black")

        if self.controller.n_centroids == 0:
            return

        for idx, centroid in enumerate(self.controller.centroids):
            start = centroid - self.centroid_radius
            end = centroid + self.centroid_radius
            self.canvas.create_oval(
                *start, *end, fill=self.controller.colors[idx], outline="black"
            )

        clusters = self.controller.get_clusters()

        for idx, cluster in enumerate(clusters):
            x0 = self.controller.centroids[idx][0]
            y0 = self.controller.centroids[idx][1]
            for point in cluster:
                x1 = point[0]
                y1 = point[1]
                self.canvas.create_line(
                    x0,
                    y0,
                    x1,
                    y1,
                    fill=self.controller.colors[idx],
                    width=self.line_width,
                )

        for idx, cluster in enumerate(clusters):
            x0 = self.controller.centroids[idx][0]
            y0 = self.controller.centroids[idx][1]
            np_cluster = np.array(cluster)

            if len(np_cluster) == 0:
                continue

            x_min, y_min = np_cluster.min(axis=0)
            x_max, y_max = np_cluster.max(axis=0)

            x_min = min(x_min, x0)
            x_max = max(x_max, x0)
            y_min = min(y_min, y0)
            y_max = max(y_max, y0)

            self.canvas.create_rectangle(
                x_min,
                y_min,
                x_max,
                y_max,
                outline=self.controller.colors[idx],
                width=self.bounding_box_width,
            )

            area = (x_max - x_min) * (y_max - y_min)
            density = (len(cluster) / area) if area > 0 else 0

            self.canvas.create_text(
                x_min,
                y_min - 5,
                text=f"Density: {density:.5f}",
                anchor="sw",
                fill=self.controller.colors[idx],
            )

    def run(self):
        self.root.mainloop()

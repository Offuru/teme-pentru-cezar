from typing import Any

import numpy as np


class Controller:
    def __init__(
        self, max_x: int, max_y: int, n_points: int = 100, max_iter: int = 100
    ):
        self.n_points = n_points
        self.n_centroids = 0
        self.max_iter = max_iter
        self.max_x = max_x
        self.max_y = max_y
        self.points = self._generate_points()
        self.centroids = np.ndarray((0, 2))
        self.labels = np.zeros(n_points, dtype=int)
        self.colors = []

    def _generate_points(self):
        x = np.random.randint(50, self.max_x, (self.n_points, 1))
        y = np.random.randint(50, self.max_y, (self.n_points, 1))

        return np.hstack((x, y))

    def _random_color(self) -> str:
        return f"#{np.random.randint(0, 0xFFFFFF):06x}"

    def add_centroid(self, position: tuple[int, int]):
        self.n_centroids += 1
        self.centroids = np.concatenate(
            (self.centroids, np.array(position).reshape((1, 2))), axis=0
        )
        self.colors.append(self._random_color())
        self._assign_clusters()

    def remove_centroid(self, idx):
        self.n_centroids -= 1
        self.centroids = np.delete(self.centroids, idx, axis=0)
        self.colors.pop(idx)
        self._assign_clusters()

    def _assign_clusters(self):
        if self.n_centroids == 0:
            return

        for idx, point in enumerate(self.points):
            distances = np.linalg.norm(self.centroids - point, axis=1)
            self.labels[idx] = np.argmin(distances)

    def _update_centroids(self) -> bool:
        new_centroids = np.zeros_like(self.centroids)

        for k in range(self.n_centroids):
            cluster_points = self.points[self.labels == k]

            if len(cluster_points) > 0:
                new_centroids[k] = cluster_points.mean(axis=0)
            else:
                new_centroids[k] = self.centroids[k]

        convergence = np.allclose(self.centroids, new_centroids)
        self.centroids = new_centroids
        return convergence

    def run_kmeans(self):
        if self.n_centroids == 0:
            return

        for _ in range(self.max_iter):
            self._assign_clusters()
            if self._update_centroids():
                break

    def get_clusters(self) -> list:
        if self.n_centroids == 0:
            return []

        clusters = [[] for _ in range(self.n_centroids)]
        for idx, label in enumerate(self.labels):
            clusters[label].append(self.points[idx])

        return clusters

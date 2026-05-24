import numpy as np


class Controller:
    def __init__(self, max_x: int, max_y: int, n_points: int = 100):
        self.n_points = n_points
        self.max_x = max_x
        self.max_y = max_y
        x = np.random.randint(50, self.max_x, (self.n_points, 1))
        y = np.random.randint(50, self.max_y, (self.n_points, 1))
        self.points = np.hstack((x, y))
        self.labels = np.random.choice([-1, 1], size=self.n_points)

    def add_point(self, position: tuple[int, int], k: int, weighted: bool = True):
        self.n_points += 1
        point = np.array([position[0], position[1]]).reshape(1, 2)
        self.labels = np.concatenate(
            (self.labels, self._get_class(point, k, weighted)), axis=0
        )
        self.points = np.concatenate(
            (self.points, point),
            axis=0,
        )

    def _get_class(self, point: np.ndarray, k: int, weighted: bool):
        mask = ~np.all(self.points == point, axis=1)
        distances = np.linalg.norm(self.points[mask] - point, axis=1)

        sorted_indices = np.argsort(distances)[:k]

        predicted_class = 1

        if not weighted:
            predicted_class = 1 if self.labels[sorted_indices].mean() > 0 else -1
        else:
            predicted_class = (
                1
                if self.labels[sorted_indices] @ (distances[sorted_indices] ** -2) > 0
                else -1
            )

        return np.array([predicted_class])

    def cross_validation(self, max_k: int, weighted: bool):

        best_k = 0
        best_acc = 0

        for k in range(1, max_k + 1):
            correct_predictions = 0
            for i in range(self.n_points):
                point = self.points[i].reshape(1, 2)
                true_label = self.labels[i]
                predicted_label = self._get_class(point, k, weighted)
                if predicted_label == true_label:
                    correct_predictions += 1
            accuracy = correct_predictions / self.n_points
            if accuracy > best_acc:
                best_acc = accuracy
                best_k = k

        return best_k, best_acc

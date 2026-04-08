import tkinter as tk
from tkinter import Canvas


class NodeWidget:
    """Visual representation of a graph node."""

    def __init__(
        self,
        canvas: Canvas,
        node_id: int,
        position: tuple[int, int],
        radius: int,
        default_color: str = "#4ECDC4",
    ):
        self.canvas = canvas
        self.node_id = node_id
        self.position = position
        self.radius = radius
        self.default_color = default_color
        self.current_color = default_color

        self.circle_id = self._draw_circle()
        self.text_id = self._draw_text()

    def _draw_circle(self) -> int:
        """Draw the node circle."""
        x, y = self.position
        return self.canvas.create_oval(
            x - self.radius,
            y - self.radius,
            x + self.radius,
            y + self.radius,
            fill=self.current_color,
            outline="#2C3E50",
            width=2,
            tags=f"node_{self.node_id}",
        )

    def _draw_text(self) -> int:
        """Draw the node label."""
        x, y = self.position
        return self.canvas.create_text(
            x,
            y,
            text=str(self.node_id),
            font=("Arial", 12, "bold"),
            fill="white",
            tags=f"node_{self.node_id}",
        )

    def update_position(self, new_position: tuple[int, int]):
        """Update the node's position."""
        self.position = new_position
        x, y = new_position

        self.canvas.coords(
            self.circle_id,
            x - self.radius,
            y - self.radius,
            x + self.radius,
            y + self.radius,
        )

        self.canvas.coords(self.text_id, x, y)

    def set_color(self, color: str):
        """Change the node's color."""
        self.current_color = color
        self.canvas.itemconfig(self.circle_id, fill=color)

    def set_label(self, label: str):
        """Change the node's label text."""
        self.canvas.itemconfig(self.text_id, text=label)

    def contains_point(self, x: int, y: int) -> bool:
        """Check if a point is inside this node."""
        distance = ((x - self.position[0]) ** 2 + (y - self.position[1]) ** 2) ** 0.5
        return distance <= self.radius

    def delete(self):
        """Remove the node from canvas."""
        self.canvas.delete(self.circle_id)
        self.canvas.delete(self.text_id)

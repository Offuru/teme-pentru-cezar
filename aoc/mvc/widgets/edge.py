from tkinter import Canvas
from .node import NodeWidget
import math


class EdgeWidget:
    """Visual representation of a graph edge."""

    def __init__(
        self,
        canvas: Canvas,
        start_node: NodeWidget,
        end_node: NodeWidget,
        capacity: int,
        flow: int,
        edge_color: str = "#2C3E50",
        label_bg: str = "#FFFBEA",
        label_border: str = "#2C3E50",
    ):
        self.canvas = canvas
        self.start_node = start_node
        self.end_node = end_node
        self.capacity = capacity
        self.flow = flow
        self.edge_color = edge_color
        self.current_color = edge_color
        self.label_bg = label_bg
        self.label_border = label_border

        self.line_id = self._draw_arc()
        self.arrow_id = self._draw_arrow()
        self.label_bg_id, self.label_id = self._draw_label()

    def _calculate_arc_points(self, num_points: int = 50) -> list[float]:
        """Calculate points along a curved arc between start and end nodes."""
        start_pos = self.start_node.position
        end_pos = self.end_node.position

        mid_x = (start_pos[0] + end_pos[0]) / 2
        mid_y = (start_pos[1] + end_pos[1]) / 2

        dx = end_pos[0] - start_pos[0]
        dy = end_pos[1] - start_pos[1]
        dist = math.sqrt(dx**2 + dy**2)

        if dist < 0.01:
            return [start_pos[0], start_pos[1], end_pos[0], end_pos[1]]

        curve_offset = -dist * 0.2

        perp_x = -dy / dist
        perp_y = dx / dist

        ctrl_x = mid_x + perp_x * curve_offset
        ctrl_y = mid_y + perp_y * curve_offset

        points = []
        for i in range(num_points + 1):
            t = i / num_points

            x = (
                (1 - t) ** 2 * start_pos[0]
                + 2 * (1 - t) * t * ctrl_x
                + t**2 * end_pos[0]
            )
            y = (
                (1 - t) ** 2 * start_pos[1]
                + 2 * (1 - t) * t * ctrl_y
                + t**2 * end_pos[1]
            )
            points.extend([x, y])

        return points

    def _get_arc_endpoint(self) -> tuple[float, float]:
        """Get the point where the arc meets the target node boundary."""

        points = self._calculate_arc_points(50)

        end_pos = self.end_node.position

        end_x, end_y = points[-2], points[-1]
        prev_x, prev_y = points[-4], points[-3]

        dx = end_x - prev_x
        dy = end_y - prev_y
        dist = math.sqrt(dx**2 + dy**2)

        if dist < 0.01:

            dx = end_pos[0] - self.start_node.position[0]
            dy = end_pos[1] - self.start_node.position[1]
            dist = math.sqrt(dx**2 + dy**2)
            if dist < 0.01:
                return end_pos

        dx /= dist
        dy /= dist

        boundary_x = end_pos[0] - dx * self.end_node.radius
        boundary_y = end_pos[1] - dy * self.end_node.radius

        return (boundary_x, boundary_y)

    def _draw_arc(self) -> int:
        """Draw the curved edge line."""
        points = self._calculate_arc_points(50)

        boundary_x, boundary_y = self._get_arc_endpoint()

        end_pos = self.end_node.position

        trimmed_points = []
        for i in range(0, len(points) - 2, 2):
            x, y = points[i], points[i + 1]
            dist_to_center = math.sqrt((x - end_pos[0]) ** 2 + (y - end_pos[1]) ** 2)

            if dist_to_center > self.end_node.radius * 1.1:
                trimmed_points.extend([x, y])
            else:

                trimmed_points.extend([boundary_x, boundary_y])
                break

        if len(trimmed_points) < 4:
            trimmed_points = points[:-2] + [boundary_x, boundary_y]

        line_id = self.canvas.create_line(
            *trimmed_points,
            width=3,
            fill=self.current_color,
            smooth=True,
            tags=f"edge_{self.start_node.node_id}_{self.end_node.node_id}",
        )

        self.canvas.tag_lower(line_id)

        return line_id

    def _draw_arrow(self) -> int:
        """Draw an arrowhead at the end of the arc."""
        boundary_x, boundary_y = self._get_arc_endpoint()

        points = self._calculate_arc_points(50)

        end_pos = self.end_node.position
        prev_x, prev_y = points[-4], points[-3]

        for i in range(len(points) - 4, 0, -2):
            x, y = points[i], points[i + 1]
            dist_to_center = math.sqrt((x - end_pos[0]) ** 2 + (y - end_pos[1]) ** 2)
            if dist_to_center > self.end_node.radius * 1.2:
                prev_x, prev_y = x, y
                break

        dx = boundary_x - prev_x
        dy = boundary_y - prev_y
        dist = math.sqrt(dx**2 + dy**2)

        if dist < 0.01:
            return None

        dx /= dist
        dy /= dist

        arrow_length = 12
        arrow_width = 6

        back_x = boundary_x - dx * arrow_length
        back_y = boundary_y - dy * arrow_length

        perp_x = -dy
        perp_y = dx

        arrow_points = [
            boundary_x,
            boundary_y,
            back_x + perp_x * arrow_width,
            back_y + perp_y * arrow_width,
            back_x - perp_x * arrow_width,
            back_y - perp_y * arrow_width,
        ]

        arrow_id = self.canvas.create_polygon(
            *arrow_points,
            fill=self.current_color,
            outline=self.current_color,
            tags=f"edge_{self.start_node.node_id}_{self.end_node.node_id}",
        )

        return arrow_id

    def _calculate_label_position(self) -> tuple[float, float]:
        """Calculate the position for the edge label at the arc's midpoint."""

        points = self._calculate_arc_points(50)

        total_points = len(points) // 2
        mid_index = (total_points // 2) * 2

        label_x = points[mid_index]
        label_y = points[mid_index + 1]

        if mid_index >= 2 and mid_index < len(points) - 2:
            prev_x = points[mid_index - 2]
            prev_y = points[mid_index - 1]
            next_x = points[mid_index + 2]
            next_y = points[mid_index + 3]

            dx = next_x - prev_x
            dy = next_y - prev_y
            dist = math.sqrt(dx**2 + dy**2)

            if dist > 0.01:

                perp_x = -dy / dist
                perp_y = dx / dist

                start_pos = self.start_node.position
                end_pos = self.end_node.position
                line_center_x = (start_pos[0] + end_pos[0]) / 2
                line_center_y = (start_pos[1] + end_pos[1]) / 2

                to_arc_x = label_x - line_center_x
                to_arc_y = label_y - line_center_y

                cross = perp_x * to_arc_x + perp_y * to_arc_y

                if cross < 0:
                    perp_x = -perp_x
                    perp_y = -perp_y

                offset = 20
                label_x += perp_x * offset
                label_y += perp_y * offset

        return (label_x, label_y)

    def _draw_label(self) -> tuple[int, int]:
        """Draw the flow/capacity label."""
        label_x, label_y = self._calculate_label_position()

        label_text = f"{self.flow}/{self.capacity}"
        padding = 8
        text_width = len(label_text) * 7

        bg_id = self.canvas.create_rectangle(
            label_x - text_width // 2 - padding,
            label_y - 12,
            label_x + text_width // 2 + padding,
            label_y + 12,
            fill=self.label_bg,
            outline=self.label_border,
            width=2,
            tags=f"edge_{self.start_node.node_id}_{self.end_node.node_id}",
        )

        text_id = self.canvas.create_text(
            label_x,
            label_y,
            text=label_text,
            font=("Arial", 11, "bold"),
            fill=self.edge_color,
            tags=f"edge_{self.start_node.node_id}_{self.end_node.node_id}",
        )

        return (bg_id, text_id)

    def set_color(self, color: str):
        """Change the edge's color."""
        self.current_color = color
        self.canvas.itemconfig(self.line_id, fill=color)
        if self.arrow_id:
            self.canvas.itemconfig(self.arrow_id, fill=color, outline=color)

    def set_label(self, label: str):
        """Change the edge's label text."""
        self.canvas.itemconfig(self.label_id, text=label)

        label_x, label_y = self._calculate_label_position()
        padding = 8
        text_width = len(label) * 7

        self.canvas.coords(
            self.label_bg_id,
            label_x - text_width // 2 - padding,
            label_y - 12,
            label_x + text_width // 2 + padding,
            label_y + 12,
        )

    def update(self):
        """Update the edge visual after node movement."""

        points = self._calculate_arc_points(50)
        boundary_x, boundary_y = self._get_arc_endpoint()

        end_pos = self.end_node.position
        trimmed_points = []
        for i in range(0, len(points) - 2, 2):
            x, y = points[i], points[i + 1]
            dist_to_center = math.sqrt((x - end_pos[0]) ** 2 + (y - end_pos[1]) ** 2)

            if dist_to_center > self.end_node.radius * 1.1:
                trimmed_points.extend([x, y])
            else:
                trimmed_points.extend([boundary_x, boundary_y])
                break

        if len(trimmed_points) < 4:
            trimmed_points = points[:-2] + [boundary_x, boundary_y]

        self.canvas.coords(self.line_id, *trimmed_points)

        if self.arrow_id:
            self.canvas.delete(self.arrow_id)
            self.arrow_id = self._draw_arrow()

        label_x, label_y = self._calculate_label_position()
        label_text = self.canvas.itemcget(self.label_id, "text")
        padding = 8
        text_width = len(label_text) * 7

        self.canvas.coords(
            self.label_bg_id,
            label_x - text_width // 2 - padding,
            label_y - 12,
            label_x + text_width // 2 + padding,
            label_y + 12,
        )
        self.canvas.coords(self.label_id, label_x, label_y)

    def delete(self):
        """Remove the edge from canvas."""
        self.canvas.delete(self.line_id)
        if self.arrow_id:
            self.canvas.delete(self.arrow_id)
        self.canvas.delete(self.label_bg_id)
        self.canvas.delete(self.label_id)

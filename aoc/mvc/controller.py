import json
import math
import random
import time

from mvc.algorithms import (
    AhujaOrlinMixin,
    AhujaOrlinShortestPathMixin,
    AhujaOrlinLayeredMixin,
    EdmondsKarpMixin,
    FordFulkersonMixin,
    GabowMixin,
    GenericMaxFlowMixin,
    PrefluxMixin,
    PrefluxQueueMixin,
)
from mvc.model import Graph

Point2D = tuple[int, int]


class Controller(
    FordFulkersonMixin,
    EdmondsKarpMixin,
    AhujaOrlinMixin,
    AhujaOrlinShortestPathMixin,
    AhujaOrlinLayeredMixin,
    GabowMixin,
    GenericMaxFlowMixin,
    PrefluxMixin,
    PrefluxQueueMixin,
):
    def __init__(self, graph: Graph, node_radius: int = 20):
        self.graph = graph
        self.node_positions = {}
        self.node_radius = node_radius
        self.view = None

    def add_node(self, position: tuple[int, int]) -> tuple[int, int] | None:
        """Add a new node at the specified position."""
        # Find the next available node ID
        if self.node_positions:
            node_id = max(self.node_positions.keys()) + 1
        else:
            node_id = 1

        # Check for collisions with existing nodes
        for existing_pos in self.node_positions.values():
            if existing_pos is None:
                continue
            distance = (
                (position[0] - existing_pos[0]) ** 2
                + (position[1] - existing_pos[1]) ** 2
            ) ** 0.5
            if distance < 2 * self.node_radius:
                return None

        # Add node to graph model
        self.graph.add_node(node_id)

        # Store position
        self.node_positions[node_id] = position

        return position

    def add_edge(self, start: int, end: int, capacity: int, flow: int = 0):
        self.graph.add_edge(start, end, capacity, flow)

    def save_graph(self, file_path: str):
        """Save graph with all data (nodes, edges, positions, flows, capacities) to JSON."""
        data = {
            "nodes": [
                {"id": node_id, "x": pos[0], "y": pos[1]}
                for node_id, pos in self.node_positions.items()
            ],
            "edges": [
                {
                    "start": edge.start,
                    "end": edge.end,
                    "capacity": edge.capacity,
                    "flow": edge.flow,
                }
                for edges in self.graph.edges.values()
                for edge in edges
            ],
        }
        with open(file_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Graph saved to {file_path}")

    def load_graph(self, file_path: str):
        """Load graph with all data from JSON file."""
        with open(file_path, "r") as f:
            data = json.load(f)

        # Clear current graph
        self.graph.edges.clear()
        self.node_positions.clear()

        # Load nodes with positions
        for node_data in data.get("nodes", []):
            node_id = node_data["id"]
            self.node_positions[node_id] = (node_data["x"], node_data["y"])
            self.graph.add_node(node_id)

        # Load edges with flows and capacities
        for edge_data in data.get("edges", []):
            self.graph.add_edge(
                edge_data["start"],
                edge_data["end"],
                edge_data["capacity"],
                edge_data.get("flow", 0),
            )

        # Refresh view if available
        if self.view:
            self.view.refresh()

        print(f"Graph loaded from {file_path}")

    def move_node(self, node: int, new_position: Point2D):
        if self._is_valid_position(node, new_position):
            self.node_positions[node] = new_position
            return new_position
        else:
            valid_position = self._find_closest_valid_position(node, new_position)
            self.node_positions[node] = valid_position
            return valid_position

    def _find_closest_valid_position(
        self, node: int, target_position: Point2D, max_iterations: int = 100
    ) -> Point2D:
        current_x, current_y = target_position
        min_distance = self._get_min_distance_between_nodes()

        for _ in range(max_iterations):
            push_vectors = []

            for other_node, other_pos in self.node_positions.items():
                if other_node == node:
                    continue

                distance = self._distance_euclidean((current_x, current_y), other_pos)

                if distance < min_distance:
                    if distance == 0:

                        angle = random.uniform(0, 2 * math.pi)
                        push_x = min_distance * math.cos(angle)
                        push_y = min_distance * math.sin(angle)
                    else:

                        direction_x = (current_x - other_pos[0]) / distance
                        direction_y = (current_y - other_pos[1]) / distance

                        violation_ratio = (min_distance - distance) / min_distance
                        push_amount = (min_distance - distance) * (1 + violation_ratio)
                        push_x = direction_x * push_amount
                        push_y = direction_y * push_amount

                    push_vectors.append((push_x, push_y))

            if not push_vectors:
                break

            avg_push_x = sum(vec[0] for vec in push_vectors) / len(push_vectors)
            avg_push_y = sum(vec[1] for vec in push_vectors) / len(push_vectors)

            push_magnitude = math.sqrt(avg_push_x**2 + avg_push_y**2)

            if push_magnitude < min_distance * 0.1 and len(push_vectors) > 0:

                escape_position = self._find_escape_direction(
                    node, (current_x, current_y), min_distance
                )
                if escape_position:
                    current_x, current_y = escape_position
                    continue

            damping_factor = 0.8
            current_x += avg_push_x * damping_factor
            current_y += avg_push_y * damping_factor

        return (int(current_x), int(current_y))

    def _find_escape_direction(
        self, node: int, position: Point2D, min_distance: float
    ) -> Point2D | None:
        x, y = position
        best_direction = None
        max_clearance = 0

        for angle in [i * (2 * math.pi / 12) for i in range(12)]:
            test_x = x + min_distance * 2 * math.cos(angle)
            test_y = y + min_distance * 2 * math.sin(angle)

            min_dist_in_direction = float("inf")
            for other_node, other_pos in self.node_positions.items():
                if other_node == node:
                    continue
                dist = self._distance_euclidean((test_x, test_y), other_pos)
                min_dist_in_direction = min(min_dist_in_direction, dist)

            if min_dist_in_direction > max_clearance:
                max_clearance = min_dist_in_direction
                best_direction = (test_x, test_y)

        return best_direction

    def _is_valid_position(self, u: int, new_position: Point2D) -> bool:
        min_distance = self._get_min_distance_between_nodes()

        for other_node, other_pos in self.node_positions.items():
            if other_node == u:
                continue
            distance = self._distance_euclidean(new_position, other_pos)
            if distance < min_distance:
                return False
        return True

    def _distance_between_nodes(self, u: int, v: int) -> float:
        x1, y1 = self.node_positions[u]
        x2, y2 = self.node_positions[v]
        return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5

    def _distance_euclidean(self, pos1: Point2D, pos2: Point2D) -> float:
        x1, y1 = pos1
        x2, y2 = pos2
        return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5

    def _get_normalised_vector(self, u: int, v: int) -> Point2D:
        x1, y1 = self.node_positions[u]
        x2, y2 = self.node_positions[v]
        return (
            (x2 - x1) / self._distance_between_nodes(u, v),
            (y2 - y1) / self._distance_between_nodes(u, v),
        )

    def _get_min_distance_between_nodes(self) -> float:
        return 2 * self.node_radius + 5

    def set_view(self, view):
        """Register the view with the controller."""
        self.view = view

    def set_status_label(self, text: str):
        """Update the status bar label in the view (no-op if no view)."""
        if self.view and hasattr(self.view, "status_label"):
            self.view.status_label.config(text=text)
            self.view.root.update_idletasks()

    def set_node_color(self, node_id: int, color: str):
        """Change a node's color."""
        if self.view and node_id in self.view.nodes:
            self.view.nodes[node_id].set_color(color)

    def set_node_label(self, node_id: int, label: str):
        """Change a node's label text."""
        if self.view and node_id in self.view.nodes:
            self.view.nodes[node_id].set_label(label)

    def highlight_node(self, node_id: int, color: str = "#FFD700"):
        """Highlight a node temporarily."""
        self.set_node_color(node_id, color)

    def set_edge_color(self, start: int, end: int, color: str):
        """Change an edge's color."""
        if self.view and (start, end) in self.view.edges:
            self.view.edges[(start, end)].set_color(color)

    def set_edge_label(self, start: int, end: int, label: str):
        """Change an edge's label text."""
        if self.view and (start, end) in self.view.edges:
            self.view.edges[(start, end)].set_label(label)

    def highlight_edge(self, start: int, end: int, color: str = "#FFD700"):
        """Highlight an edge temporarily."""
        self.set_edge_color(start, end, color)

    def reset_colors(self):
        """Reset all nodes and edges to default colors."""
        if self.view:
            for node_visual in self.view.nodes.values():
                node_visual.set_color(self.view.default_color)
            for edge_visual in self.view.edges.values():
                edge_visual.set_color(self.view.edge_color)

    def animate_step(self, delay_ms: int = 500):
        """Pause execution for visualization with actual waiting."""
        if self.view:
            self.view.root.update()
            start_time = time.time()
            while (time.time() - start_time) * 1000 < delay_ms:
                self.view.root.update()
                time.sleep(0.01)

    def update_edge_labels(self, show_residual: bool = False):
        """Update all edge labels to show current flow/capacity and optionally residual."""
        if self.view:
            for (start, end), edge_visual in self.view.edges.items():
                # Find the edge in the graph
                for edge in self.graph.edges.get(start, []):
                    if edge.end == end:
                        residual = edge.capacity - edge.flow
                        if show_residual:
                            edge_visual.set_label(
                                f"{edge.flow}/{edge.capacity} (r:{residual})"
                            )
                        else:
                            edge_visual.set_label(f"{edge.flow}/{edge.capacity}")
                        edge_visual.flow = edge.flow
                        edge_visual.capacity = edge.capacity
                        break

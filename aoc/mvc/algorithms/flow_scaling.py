from __future__ import annotations

from collections import deque

from mvc.algorithms.residual import ResidualGraphMixin


class FlowScalingMixin(ResidualGraphMixin):
    """
    Mixin implementing the excess-scaling preflow algorithm.

    The routine follows the requested pseudocode:
      - initialise the preflow exactly like the generic preflow variant
      - set r to the largest power of 2 not larger than the maximum capacity
      - while r >= 1:
          * while there exists a node x (excluding source/sink) with excess >= r:
              - choose the node with the smallest distance label
              - if an admissible residual arc can move flow without breaking the
                current scale, push as much as possible
              - otherwise relabel x

    Node labels show both the current distance and excess so the scaling phase
    can be followed visually.
    """

    _FS_ACTIVE_COLOR = "#FFFF00"
    _FS_RELABEL_COLOR = "#FFC300"
    _FS_EDGE_COLOR = "#FF00FF"
    _FS_PUSH_COLOR = "#FFD700"
    _FS_NEIGHBOUR_COLOR = "#87CEEB"

    def _bfs_distance_from_sink(self, sink: int) -> dict[int, int]:
        """Compute distance labels from the sink in the current residual graph."""
        if sink not in self.node_positions:
            return {}

        dist: dict[int, int] = {node: float("inf") for node in self.node_positions}
        queue = deque([sink])
        dist[sink] = 0

        while queue:
            u = queue.popleft()
            for v in self.node_positions:
                if dist[v] == float("inf") and self.get_residual_capacity(v, u) > 0:
                    dist[v] = dist[u] + 1
                    queue.append(v)

        return dist

    def _compute_excess(self, node: int) -> int:
        """Compute the current excess of a node: inflow minus outflow."""
        inflow = 0
        outflow = 0
        for edges in self.graph.edges.values():
            for edge in edges:
                if edge.end == node:
                    inflow += edge.flow
                if edge.start == node:
                    outflow += edge.flow
        return inflow - outflow

    def _choose_high_excess_node(
        self,
        nodes: list[int],
        source: int,
        sink: int,
        threshold: int,
        dist: dict[int, int],
    ) -> int | None:
        """Pick the active node with the smallest distance label."""
        candidates = [
            node
            for node in nodes
            if node not in (source, sink) and self._compute_excess(node) >= threshold
        ]
        if not candidates:
            return None

        return min(candidates, key=lambda node: (dist.get(node, float("inf")), node))

    def _update_node_labels(self, nodes: list[int], dist: dict[int, int]):
        """Render node labels with distance and excess information."""
        for node in nodes:
            label_dist = dist.get(node, float("inf"))
            excess = self._compute_excess(node)
            label = f"{node}\nd:{label_dist if label_dist != float('inf') else '∞'} e:{excess}"
            self.set_node_label(node, label)

    def _initial_scale(self) -> int:
        """Largest power of 2 not larger than the maximum edge capacity."""
        max_capacity = max(
            (edge.capacity for edges in self.graph.edges.values() for edge in edges),
            default=1,
        )

        scale = 1
        while scale * 2 <= max_capacity:
            scale *= 2
        return scale

    def _find_admissible_push(
        self,
        x: int,
        source: int,
        sink: int,
        threshold: int,
        dist: dict[int, int],
    ) -> tuple[int, int] | None:
        """
        Find an admissible residual arc (x, y) that can actually move flow at
        the current scale.

        Returns:
            (y, push_amount) or None if no suitable arc exists.
        """
        excess_x = self._compute_excess(x)
        if excess_x <= 0:
            return None

        admissible: list[tuple[int, int]] = []
        for y in self.node_positions:
            if dist.get(y, float("inf")) != dist.get(x, float("inf")) - 1:
                continue

            residual = self.get_residual_capacity(x, y)
            if residual <= 0:
                continue

            if y in (source, sink):
                room = excess_x
            else:
                room = threshold - self._compute_excess(y)

            push_amount = min(excess_x, residual, room)
            if push_amount > 0:
                admissible.append((y, push_amount))

        if not admissible:
            return None

        return min(
            admissible, key=lambda item: (dist.get(item[0], float("inf")), item[0])
        )

    def _push_flow(self, x: int, y: int, amount: int):
        """Push flow from x to y by updating the underlying forward/reverse edges."""
        forward_edge = next(
            (edge for edge in self.graph.edges.get(x, []) if edge.end == y), None
        )
        reverse_edge = next(
            (edge for edge in self.graph.edges.get(y, []) if edge.end == x), None
        )

        remaining = amount

        if forward_edge is not None:
            can_push = forward_edge.capacity - forward_edge.flow
            push_amount = min(remaining, can_push)
            forward_edge.flow += push_amount
            remaining -= push_amount
            print(f"[FlowScaling] Pushed {push_amount} from {x} -> {y} (forward)")

        if remaining > 0 and reverse_edge is not None:
            reverse_edge.flow -= remaining
            print(f"[FlowScaling] Pushed {remaining} from {x} -> {y} (reverse adjust)")

    def run_flow_scaling(self, source: int, sink: int, delay_ms: int = 1000):
        """
        Run the excess-scaling preflow algorithm.

        Args:
            source: source node id
            sink: sink node id
            delay_ms: animation delay in milliseconds
        """
        if source not in self.node_positions or sink not in self.node_positions:
            print(f"[FlowScaling] Invalid source ({source}) or sink ({sink})")
            return 0

        self.set_node_color(source, "#00FF00")
        self.set_node_color(sink, "#FF0000")

        dist = self._bfs_distance_from_sink(sink)

        # Initial preflow: saturate all edges out of the source.
        for edge in self.graph.edges.get(source, []):
            edge.flow = edge.capacity

        nodes = list(self.node_positions.keys())

        # Use a finite height for nodes not reached by the sink BFS so the
        # relabel steps can still make progress on source-side components.
        fallback_height = len(nodes)
        for node in nodes:
            if dist.get(node, float("inf")) == float("inf"):
                dist[node] = fallback_height

        residual = self.build_residual_graph()
        self.display_residual_graph(residual)
        self._update_node_labels(nodes, dist)
        self.set_status_label("Flow Scaling | initial preflow")
        self.animate_step(delay_ms)

        threshold = self._initial_scale()
        print(f"[FlowScaling] Initial scale r = {threshold}")

        while threshold >= 1:
            print(f"\n[FlowScaling] ══ Phase r = {threshold} ══")
            self.set_status_label(f"Flow Scaling | r = {threshold}")

            while True:
                x = self._choose_high_excess_node(nodes, source, sink, threshold, dist)
                if x is None:
                    break

                excess_x = self._compute_excess(x)
                print(
                    f"[FlowScaling] Processing node {x} with excess {excess_x} and d={dist.get(x, float('inf'))}"
                )

                if x != source and x != sink:
                    self.set_node_color(x, self._FS_ACTIVE_COLOR)
                self.animate_step(int(delay_ms * 0.25))

                chosen = self._find_admissible_push(x, source, sink, threshold, dist)

                if chosen is not None:
                    y, push_amount = chosen

                    if self.view and (x, y) in self.view.edges:
                        self.set_edge_color(x, y, self._FS_EDGE_COLOR)
                    self.animate_step(int(delay_ms * 0.25))

                    self._push_flow(x, y, push_amount)

                    residual = self.build_residual_graph()
                    self.display_residual_graph(residual)
                    self._update_node_labels(nodes, dist)
                    if self.view and (x, y) in self.view.edges:
                        self.highlight_edge(x, y, self._FS_PUSH_COLOR)
                    if y != source and y != sink:
                        self.highlight_node(y, self._FS_NEIGHBOUR_COLOR)
                    self.animate_step(delay_ms)

                    if x != source and x != sink:
                        self.set_node_color(
                            x, self.view.default_color if self.view else "#4ECDC4"
                        )
                    if self.view and (x, y) in self.view.edges:
                        self.set_edge_color(
                            x, y, self.view.edge_color if self.view else "#2C3E50"
                        )
                    self.animate_step(int(delay_ms * 0.25))

                else:
                    neighbours = [
                        y
                        for y in self.node_positions
                        if self.get_residual_capacity(x, y) > 0
                    ]

                    if neighbours:
                        min_neigh = min(dist.get(y, float("inf")) for y in neighbours)
                        new_dist = min_neigh + 1
                        old_dist = dist.get(x, float("inf"))
                        dist[x] = new_dist
                        print(
                            f"[FlowScaling] Relabel node {x} from {old_dist} to {new_dist}"
                        )

                    self._update_node_labels(nodes, dist)
                    if x != source and x != sink:
                        self.set_node_color(x, self._FS_RELABEL_COLOR)
                    self.animate_step(delay_ms // 2)

                    if x != source and x != sink:
                        self.set_node_color(
                            x, self.view.default_color if self.view else "#4ECDC4"
                        )
                    self.animate_step(int(delay_ms * 0.25))

            threshold //= 2

        total_flow = sum(edge.flow for edge in self.graph.edges.get(source, []))

        for node in nodes:
            self.set_node_label(node, str(node))

        if self.view:
            self.view.refresh()
        self.reset_colors()
        self.update_edge_labels(show_residual=False)

        self.set_status_label(f"Flow Scaling | Done | Max flow = {total_flow}")
        print(f"[FlowScaling] Maximum flow: {total_flow}")
        return total_flow

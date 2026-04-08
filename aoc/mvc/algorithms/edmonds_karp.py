from __future__ import annotations

from collections import deque

from mvc.algorithms.residual import ResidualGraphMixin


class EdmondsKarpMixin(ResidualGraphMixin):
    """
    Mixin providing the Edmonds-Karp max-flow algorithm.

    Identical to Ford-Fulkerson but uses BFS (FIFO queue) to find the
    shortest augmenting path, guaranteeing O(VE²) complexity.
    """

    # Colours used during Edmonds-Karp BFS visualisation
    _EK_CURRENT_COLOR = "#E74C3C"  # red          – node being expanded
    _EK_FRONTIER_COLOR = "#2ECC71"  # green        – node added to queue
    _EK_EDGE_EXPLORE_COLOR = "#FFC300"  # amber        – edge being considered
    _EK_EDGE_ADDED_COLOR = "#2ECC71"  # green        – edge leading to new node

    def find_augmenting_path_edmonds_karp(
        self,
        source: int,
        sink: int,
        delay_ms: int = 0,
    ) -> list[tuple[int, int]] | None:
        """
        Find the shortest augmenting path from source to sink using BFS.

        Algorithm:
          1. queue = deque([source]), prev = {source: None}
          2. Dequeue the front node (FIFO)
          3. For each neighbour with positive residual capacity not yet visited:
               - enqueue it
               - set prev[neighbour] = current
          4. Repeat until sink is reached or queue is empty.

        When delay_ms > 0 the method animates each step:
          - Red    : node currently being expanded
          - Amber  : edge being checked for residual capacity
          - Green  : edge/node added to the queue

        Returns a list of (u, v) edge tuples representing the path, or None.
        """
        if source not in self.node_positions:
            return None

        animate = delay_ms > 0

        queue: deque[int] = deque([source])
        prev: dict[int, int | None] = {source: None}

        # Mark source visually
        if animate:
            self.set_node_color(source, self._EK_FRONTIER_COLOR)
            self.animate_step(delay_ms // 2)

        while queue:
            # Always dequeue from the front (BFS)
            current = queue.popleft()

            if animate:
                if current != source and current != sink:
                    self.set_node_color(current, self._EK_CURRENT_COLOR)
                self.animate_step(delay_ms // 2)

            if current == sink:
                # Reconstruct path from prev pointers
                path: list[tuple[int, int]] = []
                node = sink
                while prev[node] is not None:
                    path.append((prev[node], node))
                    node = prev[node]
                path.reverse()
                return path

            # Expand neighbours with positive residual capacity not yet visited
            for neighbour in self.node_positions:
                if animate:
                    if (current, neighbour) in (self.view.edges if self.view else {}):
                        self.set_edge_color(
                            current, neighbour, self._EK_EDGE_EXPLORE_COLOR
                        )
                        self.animate_step(delay_ms // 4)

                if (
                    neighbour not in prev
                    and self.get_residual_capacity(current, neighbour) > 0
                ):
                    prev[neighbour] = current
                    queue.append(neighbour)

                    if animate:
                        if neighbour != source and neighbour != sink:
                            self.set_node_color(neighbour, self._EK_FRONTIER_COLOR)
                        if (current, neighbour) in (
                            self.view.edges if self.view else {}
                        ):
                            self.set_edge_color(
                                current, neighbour, self._EK_EDGE_ADDED_COLOR
                            )
                        self.animate_step(delay_ms // 2)
                elif animate:
                    if (current, neighbour) in (self.view.edges if self.view else {}):
                        self.set_edge_color(
                            current,
                            neighbour,
                            self.view.edge_color if self.view else "#2C3E50",
                        )

        return None

    def run_edmonds_karp(self, source: int, sink: int, delay_ms: int = 1000):
        """
        Run Edmonds-Karp max-flow using BFS to find shortest augmenting paths.
        Visualises each step including the residual network.

        Args:
            source: The source node ID
            sink: The sink node ID
            delay_ms: Delay in milliseconds between visualisation steps
        """
        if source not in self.node_positions or sink not in self.node_positions:
            print(f"Invalid source ({source}) or sink ({sink})")
            return 0

        total_flow = 0
        iteration = 0

        # Highlight source and sink
        self.set_node_color(source, "#00FF00")
        self.set_node_color(sink, "#FF0000")

        # Show initial residual graph
        residual = self.build_residual_graph()
        self.display_residual_graph(residual)
        print(f"[Edmonds-Karp] Initial residual graph: {residual}")
        self.animate_step(delay_ms)

        while True:
            iteration += 1
            print(f"\n[Edmonds-Karp] --- Iteration {iteration} ---")

            residual = self.build_residual_graph()

            self.reset_colors()
            self.set_node_color(source, "#00FF00")
            self.set_node_color(sink, "#FF0000")
            self.display_residual_graph(residual)
            self.animate_step(delay_ms)

            # Find shortest augmenting path via BFS
            path = self.find_augmenting_path_edmonds_karp(
                source, sink, delay_ms=delay_ms
            )

            # Restore source/sink colours that animation may have overwritten
            self.set_node_color(source, "#00FF00")
            self.set_node_color(sink, "#FF0000")
            self.animate_step(delay_ms // 2)

            if path is None:
                print("[Edmonds-Karp] No augmenting path found – algorithm complete")
                break

            print(f"[Edmonds-Karp] Found path: {path}")

            min_residual = float("inf")
            for u, v in path:
                if self.view and (u, v) in self.view.edges:
                    self.highlight_edge(u, v, "#FFD700")
                if u != source and u != sink:
                    self.highlight_node(u, "#87CEEB")
                if v != source and v != sink:
                    self.highlight_node(v, "#87CEEB")
                edge_residual = residual.get((u, v), 0)
                min_residual = min(min_residual, edge_residual)
                self.animate_step(delay_ms)

            print(f"[Edmonds-Karp] Bottleneck: {min_residual}")

            # Augment flow along the path
            for u, v in path:
                forward_edge = next(
                    (e for e in self.graph.edges.get(u, []) if e.end == v), None
                )
                reverse_edge = next(
                    (e for e in self.graph.edges.get(v, []) if e.end == u), None
                )

                remaining = min_residual

                if forward_edge is not None:
                    can_push = forward_edge.capacity - forward_edge.flow
                    push_amount = min(remaining, can_push)
                    forward_edge.flow += push_amount
                    remaining -= push_amount
                    print(
                        f"  Edge ({u},{v}): pushed {push_amount}, now {forward_edge.flow}/{forward_edge.capacity}"
                    )

                if remaining > 0 and reverse_edge is not None:
                    reverse_edge.flow -= remaining
                    print(
                        f"  Reverse edge ({v},{u}): reduced by {remaining}, now {reverse_edge.flow}/{reverse_edge.capacity}"
                    )

            total_flow += min_residual
            print(f"[Edmonds-Karp] Total flow so far: {total_flow}")

            self.reset_colors()
            self.set_node_color(source, "#00FF00")
            self.set_node_color(sink, "#FF0000")
            residual = self.build_residual_graph()
            self.display_residual_graph(residual)
            self.animate_step(delay_ms * 2)

        # Restore original graph view
        if self.view:
            self.view.refresh()
        self.reset_colors()
        self.update_edge_labels(show_residual=False)

        print(f"\n[Edmonds-Karp] Maximum flow: {total_flow}")
        return total_flow

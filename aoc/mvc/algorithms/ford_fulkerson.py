from __future__ import annotations

import random

from mvc.algorithms.residual import ResidualGraphMixin


class FordFulkersonMixin(ResidualGraphMixin):
    """Mixin providing the Ford-Fulkerson max-flow algorithm."""

    # Colours used during Ford-Fulkerson neighbour-selection visualisation
    _FF_CURRENT_COLOR = "#FF8C00"  # deep orange  – node being expanded
    _FF_FRONTIER_COLOR = "#9B59B6"  # purple       – node added to frontier
    _FF_EDGE_EXPLORE_COLOR = "#FFC300"  # amber        – edge being considered
    _FF_EDGE_ADDED_COLOR = "#9B59B6"  # purple       – edge leading to new node

    def find_augmenting_path_ford_fulkerson(
        self,
        source: int,
        sink: int,
        delay_ms: int = 0,
    ) -> list[tuple[int, int]] | None:
        """
        Find an augmenting path from source to sink using a random list-based exploration.

        Algorithm:
          1. frontier = [source], prev = {source: None}
          2. Pick a random node from frontier (remove it)
          3. For each neighbour with positive residual capacity not yet visited:
               - add to frontier
               - set prev[neighbour] = current
          4. Repeat until sink is reached or frontier is empty.

        When delay_ms > 0 the method animates each step:
          - Orange  : node currently being expanded
          - Amber   : edge being checked for residual capacity
          - Purple  : edge/node added to the frontier

        Returns a list of (u, v) edge tuples representing the path, or None.
        """
        if source not in self.node_positions:
            return None

        animate = delay_ms > 0

        frontier: list[int] = [source]
        prev: dict[int, int | None] = {source: None}

        # Mark source as in the frontier visually
        if animate:
            self.set_node_color(source, self._FF_FRONTIER_COLOR)
            self.animate_step(delay_ms // 2)

        while frontier:
            # Pick a random node from the list
            idx = random.randrange(len(frontier))
            current = frontier.pop(idx)

            if animate:
                # Highlight the node being expanded
                if current != source and current != sink:
                    self.set_node_color(current, self._FF_CURRENT_COLOR)
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
                    # Briefly flash the edge being examined
                    if (current, neighbour) in (self.view.edges if self.view else {}):
                        self.set_edge_color(
                            current, neighbour, self._FF_EDGE_EXPLORE_COLOR
                        )
                        self.animate_step(delay_ms // 4)

                if (
                    neighbour not in prev
                    and self.get_residual_capacity(current, neighbour) > 0
                ):
                    prev[neighbour] = current
                    frontier.append(neighbour)

                    if animate:
                        # Mark the newly discovered neighbour and the leading edge
                        if neighbour != source and neighbour != sink:
                            self.set_node_color(neighbour, self._FF_FRONTIER_COLOR)
                        if (current, neighbour) in (
                            self.view.edges if self.view else {}
                        ):
                            self.set_edge_color(
                                current, neighbour, self._FF_EDGE_ADDED_COLOR
                            )
                        self.animate_step(delay_ms // 2)
                elif animate:
                    # Restore edge colour if it wasn't added
                    if (current, neighbour) in (self.view.edges if self.view else {}):
                        self.set_edge_color(
                            current,
                            neighbour,
                            self.view.edge_color if self.view else "#2C3E50",
                        )

        return None

    def run_ford_fulkerson(self, source: int, sink: int, delay_ms: int = 1000):
        """
        Run Ford-Fulkerson max-flow using the random list-based augmenting path search.
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
        print(f"[Ford-Fulkerson] Initial residual graph: {residual}")
        self.animate_step(delay_ms)

        while True:
            iteration += 1
            print(f"\n[Ford-Fulkerson] --- Iteration {iteration} ---")

            residual = self.build_residual_graph()

            self.reset_colors()
            self.set_node_color(source, "#00FF00")
            self.set_node_color(sink, "#FF0000")
            self.display_residual_graph(residual)
            self.animate_step(delay_ms)

            # Run path search with neighbour-selection animation
            path = self.find_augmenting_path_ford_fulkerson(
                source, sink, delay_ms=delay_ms
            )

            # Restore source/sink colours that animation may have overwritten
            self.set_node_color(source, "#00FF00")
            self.set_node_color(sink, "#FF0000")
            self.animate_step(delay_ms // 2)

            if path is None:
                print("[Ford-Fulkerson] No augmenting path found – algorithm complete")
                break

            print(f"[Ford-Fulkerson] Found path: {path}")

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

            print(f"[Ford-Fulkerson] Bottleneck: {min_residual}")

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
            print(f"[Ford-Fulkerson] Total flow so far: {total_flow}")

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

        print(f"\n[Ford-Fulkerson] Maximum flow: {total_flow}")
        return total_flow

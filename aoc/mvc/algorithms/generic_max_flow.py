from __future__ import annotations

import random

from mvc.algorithms.residual import ResidualGraphMixin


class GenericMaxFlowMixin(ResidualGraphMixin):
    """
    Mixin providing the Generic Ford-Fulkerson (GENERIC-FD) max-flow algorithm.

    Uses a random DFS to find augmenting paths.
    """

    def find_augmenting_path_random(
        self, source: int, sink: int
    ) -> list[tuple[int, int]] | None:
        """
        Find an augmenting path from source to sink using random DFS.
        Returns a list of (node, node) tuples representing the path, or None if no path exists.
        """
        if source not in self.node_positions:
            return None

        all_nodes = set(self.node_positions.keys())

        def dfs(current: int, visited: set, path: list) -> list[tuple[int, int]] | None:
            if current == sink:
                return path

            # Get all neighbours with positive residual capacity
            neighbours = []
            for node in all_nodes:
                if node not in visited and node != current:
                    if self.get_residual_capacity(current, node) > 0:
                        neighbours.append(node)

            # Shuffle for random selection
            random.shuffle(neighbours)

            for neighbour in neighbours:
                visited.add(neighbour)
                result = dfs(neighbour, visited, path + [(current, neighbour)])
                if result is not None:
                    return result
                visited.remove(neighbour)  # Backtrack

            return None

        visited = {source}
        return dfs(source, visited, [])

    def run_max_flow(self, source: int, sink: int, delay_ms: int = 1000):
        """
        Run the generic maximum flow algorithm (GENERIC-FD).
        Visualizes each step including the residual network.

        Implements GENERIC-FD:
        (1) f := f0 (initial flow = 0)
        (2) Build residual graph G~(f)
        (3) While G~(f) contains augmenting path:
        (4)   Find augmenting path D~
        (5)   r(D~) := min{r(x,y) | (x,y) in D~}
        (6)   Augment flow
        (7)   Update G~(f) - add/remove edges based on residual

        Args:
            source: The source node ID
            sink: The sink node ID
            delay_ms: Delay in milliseconds between visualization steps
        """
        if source not in self.node_positions or sink not in self.node_positions:
            print(f"Invalid source ({source}) or sink ({sink})")
            return 0

        total_flow = 0
        iteration = 0

        # Highlight source and sink
        self.set_node_color(source, "#00FF00")  # Green for source
        self.set_node_color(sink, "#FF0000")  # Red for sink

        # Build and display initial residual graph
        residual = self.build_residual_graph()
        self.display_residual_graph(residual)
        print(f"Initial residual graph: {residual}")
        self.animate_step(delay_ms)

        while True:
            iteration += 1
            print(f"\n--- Iteration {iteration} ---")

            # Build residual graph G~(f)
            residual = self.build_residual_graph()

            # Reset colors and display residual graph
            self.reset_colors()
            self.set_node_color(source, "#00FF00")
            self.set_node_color(sink, "#FF0000")
            self.display_residual_graph(residual)
            self.animate_step(delay_ms)

            # Find augmenting path D~ using random DFS
            path = self.find_augmenting_path_random(source, sink)

            if path is None:
                print("No augmenting path found - algorithm complete")
                break

            print(f"Found path: {path}")

            # Highlight the path step by step and compute min residual
            min_residual = float("inf")

            for u, v in path:
                # Highlight the edge in path
                if (u, v) in self.view.edges:
                    self.highlight_edge(u, v, "#FFD700")  # Gold color

                # Highlight the nodes
                if u != source and u != sink:
                    self.highlight_node(u, "#87CEEB")  # Light blue
                if v != source and v != sink:
                    self.highlight_node(v, "#87CEEB")

                # r(D~) := min{r(x,y) | (x,y) in D~}
                edge_residual = residual.get((u, v), 0)
                min_residual = min(min_residual, edge_residual)

                self.animate_step(delay_ms)

            print(f"Minimum residual capacity: {min_residual}")

            # Augment flow along the path
            for u, v in path:
                # Check if (u,v) is a forward edge in original graph
                forward_edge = None
                for edge in self.graph.edges.get(u, []):
                    if edge.end == v:
                        forward_edge = edge
                        break

                # Check if (v,u) is a forward edge (meaning (u,v) is reverse)
                reverse_edge = None
                for edge in self.graph.edges.get(v, []):
                    if edge.end == u:
                        reverse_edge = edge
                        break

                remaining = min_residual

                # First increase flow on forward edge if it exists
                if forward_edge is not None:
                    can_push = forward_edge.capacity - forward_edge.flow
                    push_amount = min(remaining, can_push)
                    forward_edge.flow += push_amount
                    remaining -= push_amount
                    print(
                        f"  Edge ({u},{v}): pushed {push_amount} flow, now {forward_edge.flow}/{forward_edge.capacity}"
                    )

                # Then decrease flow on reverse edge if needed
                if remaining > 0 and reverse_edge is not None:
                    reverse_edge.flow -= remaining
                    print(
                        f"  Reverse edge ({v},{u}): reduced flow by {remaining}, now {reverse_edge.flow}/{reverse_edge.capacity}"
                    )

            total_flow += min_residual
            print(f"Total flow so far: {total_flow}")

            # Reset colors after path reduction
            self.reset_colors()
            self.set_node_color(source, "#00FF00")
            self.set_node_color(sink, "#FF0000")

            # Update residual graph G~(f) - this will add/remove edges
            residual = self.build_residual_graph()
            self.display_residual_graph(residual)
            print(f"Updated residual graph: {residual}")
            self.animate_step(delay_ms * 2)

        # Restore original graph view (show actual edges, not residual)
        self.view.refresh()
        self.reset_colors()
        self.update_edge_labels(show_residual=False)

        print(f"\nMaximum flow: {total_flow}")
        return total_flow

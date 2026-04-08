from __future__ import annotations


class ResidualGraphMixin:
    """Mixin providing residual-graph utilities shared by all max-flow algorithms."""

    def build_residual_graph(self) -> dict[tuple[int, int], int]:
        """
        Build the residual graph G~(f) from the current flow.
        Returns a dictionary mapping (u, v) -> residual capacity.
        """
        residual: dict[tuple[int, int], int] = {}

        for start, edges in self.graph.edges.items():
            for edge in edges:
                u, v = edge.start, edge.end

                # Forward edge: r(u,v) = c(u,v) - f(u,v)
                forward_residual = edge.capacity - edge.flow
                if forward_residual > 0:
                    residual[(u, v)] = residual.get((u, v), 0) + forward_residual

                # Reverse edge: r(v,u) = f(u,v)
                if edge.flow > 0:
                    residual[(v, u)] = residual.get((v, u), 0) + edge.flow

        return residual

    def display_residual_graph(self, residual: dict[tuple[int, int], int]):
        """Display the residual graph visually, adding/removing edges as needed."""
        if not self.view:
            return

        # Track which edges should exist in residual graph
        residual_edges = set(residual.keys())
        current_edges = set(self.view.edges.keys())

        # Remove edges with zero residual
        for edge_key in current_edges - residual_edges:
            self.view._remove_edge(edge_key[0], edge_key[1])

        # Add new residual edges with residual label
        for edge_key in residual_edges - current_edges:
            u, v = edge_key
            if u in self.view.nodes and v in self.view.nodes:
                self.view._add_edge(u, v, residual[edge_key], 0)
                # Immediately set the label to show residual format
                if edge_key in self.view.edges:
                    self.view.edges[edge_key].set_label(f"r:{residual[edge_key]}")

        # Update labels on existing edges
        for edge_key in residual_edges & current_edges:
            if edge_key in self.view.edges:
                self.view.edges[edge_key].set_label(f"r:{residual[edge_key]}")

    def get_residual_capacity(self, u: int, v: int) -> int:
        """Get residual capacity from u to v: r(u,v) = c(u,v) - f(u,v) + f(v,u)."""
        residual = 0

        # Forward edge contribution: capacity - flow
        for edge in self.graph.edges.get(u, []):
            if edge.end == v:
                residual += edge.capacity - edge.flow
                break

        # Reverse edge contribution: add flow from reverse edge (v -> u)
        for edge in self.graph.edges.get(v, []):
            if edge.end == u:
                residual += edge.flow
                break

        return residual

from __future__ import annotations

import random
from collections import deque

from mvc.algorithms.residual import ResidualGraphMixin


class PrefluxMixin(ResidualGraphMixin):
    """
    Mixin implementing a simple preflux / preflow-style generic max-flow algorithm.

    Behaviour:
      - initialise distance labels from sink (BFS)
      - saturate all edges out of the source (preflow)
      - maintain a list of nodes and their surplus (excess)
      - while there is an active node (excess != 0, excluding source/sink):
          * pick a random active node x
          * if exists admissible edge x->y with dist[y] == dist[x] - 1 and r(x,y) > 0
              push = min(excess_x, r(x,y)) forward on x->y
            else
              relabel x: dist[x] = min(dist[z] for z adjacent with r(x,z)>0) + 1

    The mixin exposes `run_preflux` which mirrors the other run_* methods.
    """

    def _bfs_distance_from_sink(self, sink: int) -> dict[int, int]:
        """Compute distance labels (shortest hop distance) from sink over nodes that exist."""
        if sink not in self.node_positions:
            return {}

        dist: dict[int, int] = {n: float("inf") for n in self.node_positions}
        q = deque([sink])
        dist[sink] = 0

        while q:
            u = q.popleft()
            for v in self.node_positions:
                # consider an undirected adjacency over residual > 0
                if dist[v] == float("inf") and self.get_residual_capacity(v, u) > 0:
                    dist[v] = dist[u] + 1
                    q.append(v)

        return dist

    def _compute_surplus(self, node: int) -> int:
        """Compute surplus = inflow - outflow for a node using current flows."""
        inflow = 0
        outflow = 0
        for u, edges in self.graph.edges.items():
            for e in edges:
                if e.end == node:
                    inflow += e.flow
                if e.start == node:
                    outflow += e.flow
        return inflow - outflow

    def run_preflux(self, source: int, sink: int, delay_ms: int = 1000):
        """Run the generic preflux/preflow-style max-flow algorithm.

        Args:
            source: source node id
            sink: sink node id
            delay_ms: animation delay (ms)
        """
        if source not in self.node_positions or sink not in self.node_positions:
            print(f"[Preflux] Invalid source ({source}) or sink ({sink})")
            return 0

        # Highlight source and sink
        self.set_node_color(source, "#00FF00")
        self.set_node_color(sink, "#FF0000")

        # Initial distance labels from sink
        dist = self._bfs_distance_from_sink(sink)

        # Initial preflow: saturate all edges out of source
        for edge in self.graph.edges.get(source, []):
            # push full capacity from source to its neighbours
            edge.flow = edge.capacity

        # Prepare node list and compute initial surplus labels
        nodes = list(self.node_positions.keys())

        def update_node_labels():
            for n in nodes:
                d = dist.get(n, float("inf"))
                surplus = self._compute_surplus(n)
                # show node id, distance and surplus
                label = f"{n}\nd:{d if d!=float('inf') else '∞'} s:{surplus}"
                self.set_node_label(n, label)

        # Display initial residual graph
        residual = self.build_residual_graph()
        self.display_residual_graph(residual)
        update_node_labels()
        self.animate_step(delay_ms)

        # Maintain active node list (exclude source and sink)
        def active_nodes_list():
            return [
                n
                for n in nodes
                if n not in (source, sink) and self._compute_surplus(n) != 0
            ]

        total_flow = 0
        iteration = 0

        while True:
            iteration += 1
            actives = active_nodes_list()
            print(f"[Preflux] Iteration {iteration}, active nodes: {actives}")
            if not actives:
                break

            # randomly pick an active node and color it yellow
            x = random.choice(actives)
            surplus_x = self._compute_surplus(x)

            # Color the active node yellow
            if x != source and x != sink:
                self.set_node_color(x, "#FFFF00")  # Yellow for active node
            self.animate_step(int(delay_ms * 0.25))  # 0.25s delay for coloring

            # find admissible edges x->y where dist[y] == dist[x]-1 and residual > 0
            admissible = [
                y
                for y in self.node_positions
                if dist.get(y, float("inf")) == dist.get(x, float("inf")) - 1
                and self.get_residual_capacity(x, y) > 0
            ]

            if admissible:
                y = random.choice(admissible)
                r_xy = self.get_residual_capacity(x, y)
                push = min(abs(surplus_x), r_xy)

                # Color the edge that will be processed
                if self.view and (x, y) in self.view.edges:
                    self.set_edge_color(x, y, "#FF00FF")  # Magenta for processing edge
                self.animate_step(int(delay_ms * 0.25))  # 0.25s delay for edge coloring

                # find forward and reverse edges
                forward_edge = next(
                    (e for e in self.graph.edges.get(x, []) if e.end == y), None
                )
                reverse_edge = next(
                    (e for e in self.graph.edges.get(y, []) if e.end == x), None
                )

                remaining = push

                if forward_edge is not None:
                    can_push = forward_edge.capacity - forward_edge.flow
                    push_amount = min(remaining, can_push)
                    forward_edge.flow += push_amount
                    remaining -= push_amount
                    print(f"[Preflux] Pushed {push_amount} from {x} -> {y} (forward)")

                if remaining > 0 and reverse_edge is not None:
                    # pushing on reverse reduces its flow
                    reverse_edge.flow -= remaining
                    print(
                        f"[Preflux] Pushed {remaining} from {x} -> {y} (reverse adjust)"
                    )

                # update visuals
                residual = self.build_residual_graph()
                self.display_residual_graph(residual)
                update_node_labels()
                if self.view and (x, y) in self.view.edges:
                    self.highlight_edge(x, y, "#FFD700")
                if y != source and y != sink:
                    self.highlight_node(y, "#87CEEB")
                self.animate_step(delay_ms)

                # Decolor active node and edge after processing
                if x != source and x != sink:
                    self.set_node_color(
                        x, self.view.default_color if self.view else "#4ECDC4"
                    )
                if self.view and (x, y) in self.view.edges:
                    self.set_edge_color(
                        x, y, self.view.edge_color if self.view else "#2C3E50"
                    )
                self.animate_step(int(delay_ms * 0.25))  # 0.25s delay for decoloring

            else:
                # relabel: dist[x] = min(dist[z] for z with residual(x,z)>0) + 1
                neighbours = [
                    z
                    for z in self.node_positions
                    if self.get_residual_capacity(x, z) > 0
                ]
                if neighbours:
                    min_neigh = min(dist.get(z, float("inf")) for z in neighbours)
                    newdist = min_neigh + 1
                    old = dist.get(x, float("inf"))
                    dist[x] = newdist
                    print(f"[Preflux] Relabel node {x} from {old} to {newdist}")

                # update visuals
                update_node_labels()
                if x != source and x != sink:
                    self.set_node_color(x, "#FFC300")  # Orange for relabel
                self.animate_step(delay_ms // 2)

                # Decolor active node after relabel
                if x != source and x != sink:
                    self.set_node_color(
                        x, self.view.default_color if self.view else "#4ECDC4"
                    )
                self.animate_step(int(delay_ms * 0.25))  # 0.25s delay for decoloring

        # Compute total flow as flow out of source
        total_flow = sum(e.flow for e in self.graph.edges.get(source, []))

        # restore node labels to simple ids and update edge labels
        for n in nodes:
            self.set_node_label(n, str(n))

        if self.view:
            self.view.refresh()
        self.reset_colors()
        self.update_edge_labels(show_residual=False)

        print(f"[Preflux] Maximum flow: {total_flow}")
        return total_flow

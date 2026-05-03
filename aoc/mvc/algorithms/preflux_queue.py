from __future__ import annotations

from collections import deque

from mvc.algorithms.residual import ResidualGraphMixin


class PrefluxQueueMixin(ResidualGraphMixin):
    """
    Queue-based variant of the preflux/preflow-style max-flow algorithm.

    Differences from the list-based version:
      - active nodes are maintained in a FIFO queue
      - initialization: adjacent nodes that receive flow from the source
        and therefore have surplus are added to the queue
      - algorithm: select the first node from the queue and process it
        repeatedly until either its distance label changes (relabel) or
        its surplus becomes zero
    """

    def _bfs_distance_from_sink(self, sink: int) -> dict[int, int]:
        if sink not in self.node_positions:
            return {}

        dist: dict[int, int] = {n: float("inf") for n in self.node_positions}
        q = deque([sink])
        dist[sink] = 0

        while q:
            u = q.popleft()
            for v in self.node_positions:
                if dist[v] == float("inf") and self.get_residual_capacity(v, u) > 0:
                    dist[v] = dist[u] + 1
                    q.append(v)

        return dist

    def _compute_surplus(self, node: int) -> int:
        inflow = 0
        outflow = 0
        for u, edges in self.graph.edges.items():
            for e in edges:
                if e.end == node:
                    inflow += e.flow
                if e.start == node:
                    outflow += e.flow
        return inflow - outflow

    def run_preflux_queue(self, source: int, sink: int, delay_ms: int = 1000):
        if source not in self.node_positions or sink not in self.node_positions:
            print(f"[PrefluxQueue] Invalid source ({source}) or sink ({sink})")
            return 0

        self.set_node_color(source, "#00FF00")
        self.set_node_color(sink, "#FF0000")

        dist = self._bfs_distance_from_sink(sink)

        # Initial preflow: saturate all edges out of source
        for edge in self.graph.edges.get(source, []):
            edge.flow = edge.capacity

        nodes = list(self.node_positions.keys())

        def update_node_labels():
            for n in nodes:
                d = dist.get(n, float("inf"))
                surplus = self._compute_surplus(n)
                label = f"{n}\nd:{d if d!=float('inf') else '∞'} s:{surplus}"
                self.set_node_label(n, label)

        residual = self.build_residual_graph()
        self.display_residual_graph(residual)
        update_node_labels()
        self.animate_step(delay_ms)

        # Initialize FIFO queue with adjacent nodes that have surplus (exclude source/sink)
        q: deque[int] = deque()
        in_queue: set[int] = set()
        for n in nodes:
            if n in (source, sink):
                continue
            if self._compute_surplus(n) > 0:
                q.append(n)
                in_queue.add(n)

        iteration = 0

        while q:
            iteration += 1
            x = q.popleft()
            in_queue.discard(x)
            surplus_x = self._compute_surplus(x)
            print(
                f"[PrefluxQueue] Iteration {iteration}, processing node {x}, surplus {surplus_x}"
            )

            # Color active node
            if x != source and x != sink:
                self.set_node_color(x, "#FFFF00")
            self.animate_step(int(delay_ms * 0.25))

            # Process x until relabel occurs or surplus disappears
            relabeled = False
            while True:
                surplus_x = self._compute_surplus(x)
                if surplus_x == 0:
                    break

                admissible = [
                    y
                    for y in self.node_positions
                    if dist.get(y, float("inf")) == dist.get(x, float("inf")) - 1
                    and self.get_residual_capacity(x, y) > 0
                ]

                if admissible:
                    # choose first admissible
                    y = admissible[0]
                    r_xy = self.get_residual_capacity(x, y)
                    push = min(abs(surplus_x), r_xy)

                    if self.view and (x, y) in self.view.edges:
                        self.set_edge_color(x, y, "#FF00FF")
                    self.animate_step(int(delay_ms * 0.25))

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
                        print(
                            f"[PrefluxQueue] Pushed {push_amount} from {x} -> {y} (forward)"
                        )

                    if remaining > 0 and reverse_edge is not None:
                        reverse_edge.flow -= remaining
                        print(
                            f"[PrefluxQueue] Pushed {remaining} from {x} -> {y} (reverse adjust)"
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

                    # If neighbour becomes active, add to queue
                    if (
                        y != source
                        and y != sink
                        and self._compute_surplus(y) > 0
                        and y not in in_queue
                    ):
                        q.append(y)
                        in_queue.add(y)

                    # Decolor active node and edge after processing this push
                    if x != source and x != sink:
                        self.set_node_color(
                            x, self.view.default_color if self.view else "#4ECDC4"
                        )
                    if self.view and (x, y) in self.view.edges:
                        self.set_edge_color(
                            x, y, self.view.edge_color if self.view else "#2C3E50"
                        )
                    self.animate_step(int(delay_ms * 0.25))

                    # continue processing x unless relabeled or surplus reaches 0
                    continue

                else:
                    # relabel x
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
                        relabeled = True
                        print(
                            f"[PrefluxQueue] Relabel node {x} from {old} to {newdist}"
                        )

                    update_node_labels()
                    if x != source and x != sink:
                        self.set_node_color(x, "#FFC300")
                    self.animate_step(delay_ms // 2)

                    if x != source and x != sink:
                        self.set_node_color(
                            x, self.view.default_color if self.view else "#4ECDC4"
                        )
                    self.animate_step(int(delay_ms * 0.25))

                    break

            # After processing x: if it still has surplus and was relabeled, re-enqueue
            surplus_x = self._compute_surplus(x)
            if surplus_x > 0 and relabeled:
                if x not in in_queue and x != source and x != sink:
                    q.append(x)
                    in_queue.add(x)

        # Compute total flow
        total_flow = sum(e.flow for e in self.graph.edges.get(source, []))

        for n in nodes:
            self.set_node_label(n, str(n))

        if self.view:
            self.view.refresh()
        self.reset_colors()
        self.update_edge_labels(show_residual=False)

        print(f"[PrefluxQueue] Maximum flow: {total_flow}")
        return total_flow

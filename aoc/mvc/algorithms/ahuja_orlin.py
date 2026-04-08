from __future__ import annotations

import math
import random

from mvc.algorithms.residual import ResidualGraphMixin


class AhujaOrlinMixin(ResidualGraphMixin):
    """
    Mixin providing the Ahuja-Orlin capacity-scaling max-flow algorithm.

    Algorithm (Capacity Scaling / GENERIC-SCALE):
      1. delta := largest power of 2 ≤ max residual capacity
      2. While delta >= 1:
           a. Build delta-residual graph: only edges with r(u,v) >= delta
           b. While an augmenting path exists in the delta-residual graph:
                - Find path using random DFS on delta-filtered edges
                - Augment flow along the path
           c. delta //= 2

    The label showing the current delta is updated in the view's status bar.
    """

    # Colours
    _AO_CURRENT_COLOR = "#8E44AD"  # deep purple  – node being expanded
    _AO_FRONTIER_COLOR = "#3498DB"  # blue         – node added to frontier
    _AO_EDGE_EXPLORE_COLOR = "#FFC300"  # amber        – edge being considered
    _AO_EDGE_ADDED_COLOR = "#3498DB"  # blue         – edge leading to new node
    _AO_PATH_COLOR = "#E67E22"  # orange       – augmenting path
    _AO_FILTERED_COLOR = "#BDC3C7"  # light grey   – excluded (below delta)

    # ------------------------------------------------------------------ helpers

    def _ao_display_filtered_residual(
        self,
        residual: dict[tuple[int, int], int],
        delta: int | float,
    ):
        """
        Show only residual edges with capacity >= delta; grey-out the rest.
        """
        if not self.view:
            return

        residual_edges = set(residual.keys())
        current_edges = set(self.view.edges.keys())

        # Remove edges that no longer exist in residual at all
        for edge_key in current_edges - residual_edges:
            self.view._remove_edge(edge_key[0], edge_key[1])

        for edge_key in residual_edges:
            u, v = edge_key
            cap = residual[edge_key]

            if edge_key not in self.view.edges:
                if u in self.view.nodes and v in self.view.nodes:
                    self.view._add_edge(u, v, cap, 0)

            if edge_key in self.view.edges:
                ev = self.view.edges[edge_key]
                if cap >= delta:
                    ev.set_color(self.view.edge_color)
                    ev.set_label(f"r:{cap}")
                else:
                    ev.set_color(self._AO_FILTERED_COLOR)
                    ev.set_label(f"r:{cap}")

    def _ao_find_path(
        self,
        source: int,
        sink: int,
        delta: int | float,
        delay_ms: int = 0,
    ) -> list[tuple[int, int]] | None:
        """
        Random DFS augmenting-path search restricted to edges with r(u,v) >= delta.

        Returns a list of (u, v) edge tuples, or None if no path exists.
        """
        if source not in self.node_positions:
            return None

        animate = delay_ms > 0
        all_nodes = list(self.node_positions.keys())

        def dfs(current: int, visited: set, path: list) -> list[tuple[int, int]] | None:
            if current == sink:
                return path

            neighbours = [
                n
                for n in all_nodes
                if n not in visited
                and n != current
                and self.get_residual_capacity(current, n) >= delta
            ]
            random.shuffle(neighbours)

            for nb in neighbours:
                if animate:
                    if (current, nb) in (self.view.edges if self.view else {}):
                        self.set_edge_color(current, nb, self._AO_EDGE_EXPLORE_COLOR)
                        self.animate_step(delay_ms // 4)

                visited.add(nb)

                if animate:
                    if nb != source and nb != sink:
                        self.set_node_color(nb, self._AO_FRONTIER_COLOR)
                    if (current, nb) in (self.view.edges if self.view else {}):
                        self.set_edge_color(current, nb, self._AO_EDGE_ADDED_COLOR)
                    self.animate_step(delay_ms // 3)

                result = dfs(nb, visited, path + [(current, nb)])
                if result is not None:
                    return result

                visited.remove(nb)  # backtrack

                if animate:
                    if nb != source and nb != sink:
                        self.set_node_color(
                            nb, self.view.default_color if self.view else "#4ECDC4"
                        )
                    if (current, nb) in (self.view.edges if self.view else {}):
                        cap = self.get_residual_capacity(current, nb)
                        color = (
                            self.view.edge_color
                            if cap >= delta
                            else self._AO_FILTERED_COLOR
                        )
                        self.set_edge_color(current, nb, color)

            return None

        if animate:
            self.set_node_color(source, self._AO_FRONTIER_COLOR)
            self.animate_step(delay_ms // 2)

        visited = {source}
        return dfs(source, visited, [])

    # ------------------------------------------------------------------ main

    def run_ahuja_orlin(self, source: int, sink: int, delay_ms: int = 1000):
        """
        Run Ahuja-Orlin capacity-scaling max-flow.

        Steps visualised:
          - Status label shows the current delta (minimum residual threshold)
          - Delta-residual graph: active edges (r >= delta) shown normally,
            inactive edges shown in light grey
          - Augmenting path highlighted in orange before augmentation
          - Delta halved when no path exists in the current delta-residual graph

        Args:
            source:   Source node ID
            sink:     Sink node ID
            delay_ms: Delay in milliseconds between visualisation steps
        """
        if source not in self.node_positions or sink not in self.node_positions:
            print(f"[Ahuja-Orlin] Invalid source ({source}) or sink ({sink})")
            return 0

        # --- initialise delta to largest power of 2 <= max capacity ----------
        max_cap = max(
            (edge.capacity for edges in self.graph.edges.values() for edge in edges),
            default=1,
        )
        delta = 1
        while delta * 2 <= max_cap:
            delta *= 2

        total_flow = 0

        self.set_node_color(source, "#00FF00")
        self.set_node_color(sink, "#FF0000")

        residual = self.build_residual_graph()
        self._ao_display_filtered_residual(residual, delta)
        self.set_status_label(f"Ahuja-Orlin  |  Δ = {delta}")
        print(f"[Ahuja-Orlin] Starting  max_cap={max_cap}  initial Δ={delta}")
        self.animate_step(delay_ms)

        while delta >= 1:
            print(f"\n[Ahuja-Orlin] ══ Phase Δ = {delta} ══")
            self.set_status_label(f"Ahuja-Orlin  |  Δ = {delta}")

            # Reset colours, redraw filtered residual for this delta phase
            self.reset_colors()
            self.set_node_color(source, "#00FF00")
            self.set_node_color(sink, "#FF0000")
            residual = self.build_residual_graph()
            self._ao_display_filtered_residual(residual, delta)
            self.animate_step(delay_ms)

            phase_iteration = 0

            while True:
                phase_iteration += 1

                # Rebuild and redisplay before each path search
                residual = self.build_residual_graph()
                self.reset_colors()
                self.set_node_color(source, "#00FF00")
                self.set_node_color(sink, "#FF0000")
                self._ao_display_filtered_residual(residual, delta)
                self.set_status_label(
                    f"Ahuja-Orlin  |  Δ = {delta}  |  path search #{phase_iteration}"
                )
                self.animate_step(delay_ms)

                # Find augmenting path in delta-residual graph
                path = self._ao_find_path(source, sink, delta, delay_ms=delay_ms)

                # Restore source/sink colours
                self.set_node_color(source, "#00FF00")
                self.set_node_color(sink, "#FF0000")
                self.animate_step(delay_ms // 2)

                if path is None:
                    print(f"[Ahuja-Orlin] No Δ-path for Δ={delta} → halving")
                    break

                print(f"[Ahuja-Orlin] Δ={delta}  path: {path}")

                # Highlight path and compute bottleneck
                min_residual: float = float("inf")
                for u, v in path:
                    if self.view and (u, v) in self.view.edges:
                        self.highlight_edge(u, v, self._AO_PATH_COLOR)
                    if u != source and u != sink:
                        self.highlight_node(u, "#87CEEB")
                    if v != source and v != sink:
                        self.highlight_node(v, "#87CEEB")
                    min_residual = min(min_residual, residual.get((u, v), 0))
                    self.animate_step(delay_ms)

                print(f"[Ahuja-Orlin] Bottleneck: {min_residual}")
                self.set_status_label(
                    f"Ahuja-Orlin  |  Δ = {delta}  |  bottleneck = {min_residual}"
                )
                self.animate_step(delay_ms // 2)

                # Augment flow along path
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
                        push = min(remaining, can_push)
                        forward_edge.flow += push
                        remaining -= push
                        print(
                            f"  ({u},{v}): pushed {push}, now {forward_edge.flow}/{forward_edge.capacity}"
                        )

                    if remaining > 0 and reverse_edge is not None:
                        reverse_edge.flow -= remaining
                        print(
                            f"  reverse ({v},{u}): reduced by {remaining}, now {reverse_edge.flow}/{reverse_edge.capacity}"
                        )

                total_flow += min_residual
                print(f"[Ahuja-Orlin] Total flow: {total_flow}")

            # Halve delta for next phase
            delta //= 2

        # --- done -------------------------------------------------------------
        if self.view:
            self.view.refresh()
        self.reset_colors()
        self.update_edge_labels(show_residual=False)
        self.set_status_label(f"Ahuja-Orlin  |  Done  |  Max flow = {total_flow}")

        print(f"\n[Ahuja-Orlin] Maximum flow: {total_flow}")
        return total_flow

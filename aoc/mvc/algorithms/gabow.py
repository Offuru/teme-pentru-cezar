from __future__ import annotations

import random

from mvc.algorithms.residual import ResidualGraphMixin


class GabowMixin(ResidualGraphMixin):
    """
    Mixin providing Gabow's bit-scaling max-flow algorithm.

    Algorithm:
      1. Save original capacities.
      2. Repeatedly halve all capacities (showing each residual network with
         a 1-second delay) until every edge has capacity ≤ 1.
      3. At the base level (all capacities ≤ 1) compute the max flow using
         random augmenting paths (same style as Generic max-flow).
      4. Move one level up: set starting flow = 2 × previous flow, display
         that starting configuration, then find augmenting paths on the
         residual network.
      5. Repeat step 4 until reaching the original network. The resulting
         flows are the max-flow solution.

    Visualisation:
      • Halving phase shown with 1 s delay per level.
      • Each level: starting (doubled) flow displayed, then augmenting
        paths animated on the residual network.
      • Final flow/capacity labels restored at the end.
    """

    # Colours
    _GB_FRONTIER_COLOR = "#1ABC9C"  # teal         – node added to frontier
    _GB_EDGE_EXPLORE_COLOR = "#FFC300"  # amber     – edge being considered
    _GB_EDGE_ADDED_COLOR = "#1ABC9C"  # teal       – edge added to path
    _GB_PATH_COLOR = "#E67E22"  # orange      – augmenting path
    _GB_ZERO_CAP_COLOR = "#ECF0F1"  # near-white  – zero-capacity edge

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _gb_display_scaled_network(self):
        """
        Show the current network.  Edges with capacity 0 are drawn nearly
        invisible; others show their flow/capacity values.
        """
        if not self.view:
            return
        for (u, v), ev in self.view.edges.items():
            for edge in self.graph.edges.get(u, []):
                if edge.end == v:
                    if edge.capacity == 0:
                        ev.set_color(self._GB_ZERO_CAP_COLOR)
                        ev.set_label("0/0")
                    else:
                        ev.set_color(self.view.edge_color)
                        ev.set_label(f"{edge.flow}/{edge.capacity}")
                    ev.flow = edge.flow
                    ev.capacity = edge.capacity
                    break

    def _gb_find_path(
        self,
        source: int,
        sink: int,
        delay_ms: int = 0,
    ) -> list[tuple[int, int]] | None:
        """
        Random DFS augmenting-path search on the current residual network.
        Identical in spirit to the Generic max-flow random path finder.
        """
        if source not in self.node_positions:
            return None

        animate = delay_ms > 0
        all_nodes = list(self.node_positions.keys())

        def dfs(current: int, visited: set, path: list):
            if current == sink:
                return path

            neighbours = [
                n
                for n in all_nodes
                if n not in visited
                and n != current
                and self.get_residual_capacity(current, n) > 0
            ]
            random.shuffle(neighbours)

            for nb in neighbours:
                if animate:
                    if (current, nb) in (self.view.edges if self.view else {}):
                        self.set_edge_color(current, nb, self._GB_EDGE_EXPLORE_COLOR)
                        self.animate_step(delay_ms // 4)

                visited.add(nb)

                if animate:
                    if nb != source and nb != sink:
                        self.set_node_color(nb, self._GB_FRONTIER_COLOR)
                    if (current, nb) in (self.view.edges if self.view else {}):
                        self.set_edge_color(current, nb, self._GB_EDGE_ADDED_COLOR)
                    self.animate_step(delay_ms // 3)

                result = dfs(nb, visited, path + [(current, nb)])
                if result is not None:
                    return result

                visited.remove(nb)  # backtrack

                if animate:
                    if nb != source and nb != sink:
                        col = self.view.default_color if self.view else "#4ECDC4"
                        self.set_node_color(nb, col)
                    if (current, nb) in (self.view.edges if self.view else {}):
                        for edge in self.graph.edges.get(current, []):
                            if edge.end == nb:
                                c = (
                                    self.view.edge_color
                                    if edge.capacity > 0
                                    else self._GB_ZERO_CAP_COLOR
                                )
                                self.set_edge_color(current, nb, c)
                                break

            return None

        if animate:
            self.set_node_color(source, self._GB_FRONTIER_COLOR)
            self.animate_step(delay_ms // 2)

        return dfs(source, {source}, [])

    def _gb_augment_along_path(
        self,
        path: list[tuple[int, int]],
        residual: dict[tuple[int, int], int],
        source: int,
        sink: int,
        delay_ms: int,
    ) -> int:
        """Highlight path, compute bottleneck, push flow. Returns pushed amount."""
        min_res: float = float("inf")
        for u, v in path:
            if self.view and (u, v) in self.view.edges:
                self.highlight_edge(u, v, self._GB_PATH_COLOR)
            if u != source and u != sink:
                self.highlight_node(u, "#87CEEB")
            if v != source and v != sink:
                self.highlight_node(v, "#87CEEB")
            min_res = min(min_res, residual.get((u, v), 0))
            self.animate_step(delay_ms)

        min_res = int(min_res)
        print(f"    bottleneck = {min_res}")

        for u, v in path:
            fwd = next((e for e in self.graph.edges.get(u, []) if e.end == v), None)
            rev = next((e for e in self.graph.edges.get(v, []) if e.end == u), None)
            remaining = min_res

            if fwd is not None:
                push = min(remaining, fwd.capacity - fwd.flow)
                fwd.flow += push
                remaining -= push

            if remaining > 0 and rev is not None:
                rev.flow -= remaining

        return min_res

    def _gb_run_augmentation_loop(
        self,
        source: int,
        sink: int,
        level_label: str,
        delay_ms: int,
    ):
        """
        Find and augment random paths until no more exist (like Generic
        max-flow).  Operates on whatever capacities/flows are currently
        set in self.graph.
        """
        phase_iter = 0

        while True:
            phase_iter += 1

            self.reset_colors()
            self.set_node_color(source, "#00FF00")
            self.set_node_color(sink, "#FF0000")

            # Show residual network before searching for a path
            residual = self.build_residual_graph()
            self.display_residual_graph(residual)

            self.set_status_label(f"{level_label}  |  path #{phase_iter}")
            self.animate_step(delay_ms)

            path = self._gb_find_path(source, sink, delay_ms=0)

            if path is None:
                print(
                    f"[Gabow]    No augmenting path after "
                    f"{phase_iter - 1} augmentation(s)"
                )
                break

            print(f"[Gabow]    path: {path}")
            pushed = self._gb_augment_along_path(path, residual, source, sink, delay_ms)
            print(f"[Gabow]    pushed {pushed}")

            # Show flux after augmentation
            self.reset_colors()
            self.set_node_color(source, "#00FF00")
            self.set_node_color(sink, "#FF0000")
            if self.view:
                self.view.refresh()
            self._gb_display_scaled_network()
            self.animate_step(delay_ms)

    # ------------------------------------------------------------------
    # main
    # ------------------------------------------------------------------

    def run_gabow(self, source: int, sink: int, delay_ms: int = 1000):
        """
        Run Gabow's bit-scaling max-flow algorithm with full visualisation.

        Steps:
          1. Repeatedly halve capacities (1 s delay each) until all ≤ 1.
          2. At base level, compute max-flow with random augmenting paths.
          3. Scale up: double flows, show starting flux, augment on residual.
          4. Repeat until the original network is restored.

        Args:
            source:   Source node ID.
            sink:     Sink node ID.
            delay_ms: Base animation delay in ms.
        """
        if source not in self.node_positions or sink not in self.node_positions:
            print(f"[Gabow] Invalid source ({source}) or sink ({sink})")
            return 0

        # ── 1. Save original capacities ────────────────────────────────────
        original_caps: dict[tuple[int, int], int] = {}
        for edges in self.graph.edges.values():
            for edge in edges:
                original_caps[(edge.start, edge.end)] = edge.capacity

        max_cap = max(original_caps.values(), default=0)
        if max_cap == 0:
            print("[Gabow] All capacities are zero – nothing to do.")
            return 0

        # ── 2. Build the sequence of divisors ──────────────────────────────
        #    Keep halving until all capacities are ≤ 1.
        #    divisors = [1, 2, 4, …, 2^(k-1)]  where 2^(k-1) is the
        #    smallest power of 2 such that  max_cap // 2^(k-1) <= 1.
        divisors: list[int] = [1]
        d = 1
        while max_cap // d > 1:
            d *= 2
            divisors.append(d)
        # divisors is now [1, 2, 4, …, largest] — we process largest first
        # when scaling down, and smallest-first when scaling up.

        k = len(divisors)
        print(f"[Gabow] max_cap={max_cap}  levels={k}  divisors={divisors}")

        # ── 3. Show halving phase (1 s delay between each) ────────────────
        self.set_node_color(source, "#00FF00")
        self.set_node_color(sink, "#FF0000")

        for level_idx, divisor in enumerate(divisors):
            self.set_status_label(
                f"Gabow  |  Halving capacities …  "
                f"Level {level_idx + 1}/{k}  (÷{divisor})"
            )
            for edges in self.graph.edges.values():
                for edge in edges:
                    edge.capacity = original_caps[(edge.start, edge.end)] // divisor
                    edge.flow = 0
            self._gb_display_scaled_network()
            self.animate_step(1000)  # 1 second between halvings

        # ── 4. Process levels from smallest (largest divisor) to original──
        #    reversed divisors: [largest, …, 2, 1]
        processing_order = list(reversed(divisors))

        for proc_idx, divisor in enumerate(processing_order):
            level_num = proc_idx + 1
            is_base = proc_idx == 0  # base level = all capacities ≤ 1

            print(f"\n[Gabow] ══ Level {level_num}/{k}  " f"divisor={divisor} ══")

            # ── a. Set scaled capacities ──────────────────────────────────
            for edges in self.graph.edges.values():
                for edge in edges:
                    edge.capacity = original_caps[(edge.start, edge.end)] // divisor

            # ── b. Set starting flows ─────────────────────────────────────
            if is_base:
                # Base level: all flows start at 0
                for edges in self.graph.edges.values():
                    for edge in edges:
                        edge.flow = 0
                print("[Gabow]    Base level – all capacities ≤ 1, flows start at 0")
            else:
                # Double flows from the previous level
                print("[Gabow]    Doubling flows from previous level")
                self.set_status_label(
                    f"Gabow  |  Level {level_num}/{k} (÷{divisor})  "
                    f"|  Starting flux = 2 × previous"
                )
                for edges in self.graph.edges.values():
                    for edge in edges:
                        doubled = edge.flow * 2
                        # Clamp to new capacity (safety guard)
                        edge.flow = min(doubled, edge.capacity)

            # ── c. Display starting configuration ─────────────────────────
            self.reset_colors()
            self.set_node_color(source, "#00FF00")
            self.set_node_color(sink, "#FF0000")
            self._gb_display_scaled_network()

            level_label = f"Gabow  |  Level {level_num}/{k} (÷{divisor})"
            if is_base:
                self.set_status_label(
                    f"{level_label}  |  Base: finding augmenting paths…"
                )
            else:
                self.set_status_label(
                    f"{level_label}  |  Doubled flows shown, "
                    f"augmenting on residual…"
                )
            self.animate_step(delay_ms)

            # ── d. Run augmenting-path loop (like Generic max-flow) ───────
            self._gb_run_augmentation_loop(source, sink, level_label, delay_ms)

        # ── 5. Compute final flow and restore display ──────────────────────
        total_flow = sum(edge.flow for edge in self.graph.edges.get(source, []))

        if self.view:
            self.view.refresh()
        self.reset_colors()
        self.update_edge_labels(show_residual=False)
        self.set_status_label(f"Gabow  |  Done  |  Max flow = {total_flow}")

        print(f"\n[Gabow] Maximum flow: {total_flow}")
        return total_flow

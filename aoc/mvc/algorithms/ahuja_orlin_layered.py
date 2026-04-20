from __future__ import annotations

import random
from collections import deque

from mvc.algorithms.residual import ResidualGraphMixin


class AhujaOrlinLayeredMixin(ResidualGraphMixin):
    """
    Ahuja-Orlin layered networks algorithm with distance labels and blocking.

    Uses BFS to compute distance labels from sink, then greedily traverses
    admissible edges while blocking nodes that have no admissible outgoing edges.
    When source becomes blocked, recompute BFS and reset blocking.
    """

    _AOLN_BFS_EDGE_COLOR = "#3498DB"  # blue
    _AOLN_PATH_EDGE_COLOR = "#E67E22"  # orange
    _AOLN_CURRENT_NODE_COLOR = "#9B59B6"  # purple
    _AOLN_BLOCKED_NODE_COLOR = "#E74C3C"  # red

    def _aoln_is_admissible_edge(
        self,
        u: int,
        v: int,
        cap: int,
        dist: dict[int, int],
        blocked: dict[int, bool],
        n: int,
    ) -> bool:
        """Layered admissibility: positive residual and exactly one level closer."""
        if cap <= 0:
            return False
        if blocked.get(v, False):
            return False
        return dist.get(u, n) == dist.get(v, n) + 1

    def _aoln_bfs_from_sink(
        self,
        sink: int,
        nodes: list[int],
        residual: dict[tuple[int, int], int],
    ) -> tuple[dict[int, int], dict[int, int | None]]:
        """
        Compute distances to sink using reverse-BFS over residual edges.

        Returns:
            dist: node -> distance to sink (n means unreachable)
            bfs_parent: node -> next node towards sink in BFS tree
        """
        n = len(nodes)
        dist = {node: n for node in nodes}
        bfs_parent: dict[int, int | None] = {node: None for node in nodes}

        dist[sink] = 0
        queue = deque([sink])

        reverse_adj: dict[int, list[int]] = {node: [] for node in nodes}
        for u, v in residual:
            reverse_adj[v].append(u)

        while queue:
            current = queue.popleft()
            for predecessor in reverse_adj.get(current, []):
                if dist[predecessor] == n:
                    dist[predecessor] = dist[current] + 1
                    bfs_parent[predecessor] = current
                    queue.append(predecessor)

        return dist, bfs_parent

    def _aoln_path_nodes(
        self,
        parent: dict[int, int | None],
        current: int,
    ) -> list[int]:
        """Return the current path as a node list."""
        path_nodes: list[int] = []
        seen: set[int] = set()
        node = current

        while node is not None and node not in seen:
            path_nodes.append(node)
            seen.add(node)
            node = parent.get(node)

        path_nodes.reverse()
        return path_nodes

    def _aoln_visualize_state(
        self,
        source: int,
        sink: int,
        residual: dict[tuple[int, int], int],
        dist: dict[int, int],
        bfs_parent: dict[int, int | None],
        blocked: dict[int, bool],
        current_path_nodes: list[int],
        current_node: int,
        delay_ms: int,
        total_flow: int,
    ):
        """Refresh residual graph + labels + BFS/current path + blocked status."""
        if not self.view:
            return

        n = len(self.node_positions)

        self.reset_colors()
        self.display_residual_graph(residual)

        self.set_node_color(source, "#00FF00")
        self.set_node_color(sink, "#FF0000")

        for node_id in self.node_positions:
            value = dist.get(node_id, n)
            if blocked.get(node_id, False):
                distance_text = "blocked"
            else:
                distance_text = "∞" if value >= n else str(value)
            self.set_node_label(node_id, f"{node_id}\nd:{distance_text}")

            if blocked.get(node_id, False) and node_id not in (source, sink):
                self.set_node_color(node_id, self._AOLN_BLOCKED_NODE_COLOR)

        for index in range(len(current_path_nodes) - 1):
            u = current_path_nodes[index]
            v = current_path_nodes[index + 1]
            if (u, v) in self.view.edges:
                self.set_edge_color(u, v, self._AOLN_PATH_EDGE_COLOR)

        if current_node in self.view.nodes and current_node not in (source, sink):
            self.set_node_color(current_node, self._AOLN_CURRENT_NODE_COLOR)

        ds = dist.get(source, n)
        ds_text = "∞" if ds >= n else str(ds)
        self.set_status_label(
            f"Ahuja-Orlin Layered | flow={total_flow} | d(source)={ds_text}"
        )
        self.animate_step(delay_ms)

    def _aoln_augment_path(
        self,
        path_edges: list[tuple[int, int]],
        residual: dict[tuple[int, int], int],
    ) -> int:
        """Augment one source-sink path and return bottleneck."""
        bottleneck = min(residual[(u, v)] for u, v in path_edges)

        for u, v in path_edges:
            forward_edge = next(
                (edge for edge in self.graph.edges.get(u, []) if edge.end == v),
                None,
            )
            reverse_edge = next(
                (edge for edge in self.graph.edges.get(v, []) if edge.end == u),
                None,
            )

            remaining = bottleneck

            if forward_edge is not None:
                can_push = forward_edge.capacity - forward_edge.flow
                push = min(remaining, can_push)
                forward_edge.flow += push
                remaining -= push

            if remaining > 0 and reverse_edge is not None:
                reverse_edge.flow -= remaining

        return bottleneck

    def run_ahuja_orlin_layered(
        self,
        source: int,
        sink: int,
        delay_ms: int = 1000,
    ) -> int:
        """
        Run Ahuja-Orlin layered networks algorithm with distance labels and blocking.

        Algorithm flow:
          1. Compute BFS distances from sink; mark all nodes as not blocked (b=1)
          2. While d(source) < n:
             a. If source not blocked:
                - Find admissible edge (d(current) = d(next) + 1, cap > 0)
                - If found: move forward, update path
                - If reached sink: augment, reset to source, recompute residual
                - If no admissible: mark current as blocked, backtrack
             b. If source is blocked: recompute BFS, reset blocking, reset to source

        Visualization:
          - Node labels: id | d=distance | blocked/ok
          - BFS tree edges: blue
          - Current augmenting path edges: orange
          - Blocked nodes: red
          - Current node: purple
          - Status: live flow and d(source)
        """
        if source not in self.node_positions or sink not in self.node_positions:
            print(f"[AOLN] Invalid source ({source}) or sink ({sink})")
            return 0

        nodes = list(self.node_positions.keys())
        n = len(nodes)
        total_flow = 0

        residual = self.build_residual_graph()
        dist, bfs_parent = self._aoln_bfs_from_sink(sink, nodes, residual)

        blocked: dict[int, bool] = {node: False for node in nodes}
        parent: dict[int, int | None] = {node: None for node in nodes}
        current = source

        self._aoln_visualize_state(
            source=source,
            sink=sink,
            residual=residual,
            dist=dist,
            bfs_parent=bfs_parent,
            blocked=blocked,
            current_path_nodes=[source],
            current_node=current,
            delay_ms=delay_ms,
            total_flow=total_flow,
        )

        while dist.get(source, n) < n:
            if not blocked.get(source, False):
                current_path_nodes = self._aoln_path_nodes(parent, current)
                current_path_set = set(current_path_nodes)

                admissible = [
                    v
                    for (u, v), cap in residual.items()
                    if u == current
                    and self._aoln_is_admissible_edge(current, v, cap, dist, blocked, n)
                    and v not in current_path_set
                ]

                if admissible:
                    next_node = random.choice(admissible)
                    parent[next_node] = current
                    current = next_node

                    current_path_nodes = self._aoln_path_nodes(parent, current)
                    self._aoln_visualize_state(
                        source=source,
                        sink=sink,
                        residual=residual,
                        dist=dist,
                        bfs_parent=bfs_parent,
                        blocked=blocked,
                        current_path_nodes=current_path_nodes,
                        current_node=current,
                        delay_ms=delay_ms,
                        total_flow=total_flow,
                    )

                    if current == sink:
                        path_edges = [
                            (
                                current_path_nodes[i],
                                current_path_nodes[i + 1],
                            )
                            for i in range(len(current_path_nodes) - 1)
                        ]

                        bottleneck = self._aoln_augment_path(path_edges, residual)
                        total_flow += bottleneck

                        residual = self.build_residual_graph()
                        parent = {node: None for node in nodes}
                        current = source

                        self._aoln_visualize_state(
                            source=source,
                            sink=sink,
                            residual=residual,
                            dist=dist,
                            bfs_parent=bfs_parent,
                            blocked=blocked,
                            current_path_nodes=[source],
                            current_node=current,
                            delay_ms=delay_ms,
                            total_flow=total_flow,
                        )
                else:
                    blocked[current] = True

                    if current != source:
                        previous = current
                        current = parent[current]
                        parent[previous] = None

                    current_path_nodes = self._aoln_path_nodes(parent, current)
                    self._aoln_visualize_state(
                        source=source,
                        sink=sink,
                        residual=residual,
                        dist=dist,
                        bfs_parent=bfs_parent,
                        blocked=blocked,
                        current_path_nodes=current_path_nodes,
                        current_node=current,
                        delay_ms=delay_ms,
                        total_flow=total_flow,
                    )
            else:
                residual = self.build_residual_graph()
                dist, bfs_parent = self._aoln_bfs_from_sink(sink, nodes, residual)
                blocked = {node: False for node in nodes}

                if dist.get(source, n) >= n:
                    blocked[source] = True

                parent = {node: None for node in nodes}
                current = source

                self._aoln_visualize_state(
                    source=source,
                    sink=sink,
                    residual=residual,
                    dist=dist,
                    bfs_parent=bfs_parent,
                    blocked=blocked,
                    current_path_nodes=[source],
                    current_node=current,
                    delay_ms=delay_ms,
                    total_flow=total_flow,
                )

        if self.view:
            self.view.refresh()
        self.reset_colors()
        self.update_edge_labels(show_residual=False)
        for node_id in self.node_positions:
            self.set_node_label(node_id, str(node_id))

        self.set_status_label(f"Ahuja-Orlin Layered | Done | Max flow = {total_flow}")
        print(f"[AOLN] Maximum flow: {total_flow}")
        return total_flow

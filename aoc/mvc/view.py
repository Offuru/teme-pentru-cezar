import tkinter as tk
from tkinter import Canvas

from mvc.controller import Controller
from .widgets.node import NodeWidget
from .widgets.edge import EdgeWidget
from .widgets.edge_dialog import EdgeDialogWidget


class GraphView:
    def __init__(self, controller: Controller, width: int = 800, height: int = 600):
        self.controller = controller
        self.controller.set_view(self)
        self.width = width
        self.height = height

        self.root = tk.Tk()
        self.root.title("Graph Visualizer")

        # Toolbar
        self.toolbar = tk.Frame(self.root, bg="#2C3E50", pady=5)
        self.toolbar.pack(side=tk.TOP, fill=tk.X)

        btn_style = {
            "bg": "#4ECDC4",
            "fg": "white",
            "relief": tk.FLAT,
            "padx": 12,
            "pady": 5,
            "font": ("Segoe UI", 9, "bold"),
            "cursor": "hand2",
            "activebackground": "#45B7AA",
            "activeforeground": "white",
        }

        tk.Button(
            self.toolbar,
            text="Max Flow (F)",
            command=lambda: self._on_max_flow_key(None),
            **btn_style,
        ).pack(side=tk.LEFT, padx=(8, 4))

        tk.Button(
            self.toolbar,
            text="Ford-Fulkerson",
            command=self._on_ford_fulkerson,
            **btn_style,
        ).pack(side=tk.LEFT, padx=4)

        tk.Button(
            self.toolbar,
            text="Edmonds-Karp",
            command=self._on_edmonds_karp,
            **btn_style,
        ).pack(side=tk.LEFT, padx=4)

        tk.Button(
            self.toolbar,
            text="Ahuja-Orlin",
            command=self._on_ahuja_orlin,
            **btn_style,
        ).pack(side=tk.LEFT, padx=4)

        tk.Button(
            self.toolbar,
            text="Ahuja-Orlin SP",
            command=self._on_ahuja_orlin_shortest_path,
            **btn_style,
        ).pack(side=tk.LEFT, padx=4)

        tk.Button(
            self.toolbar,
            text="Ahuja-Orlin LN",
            command=self._on_ahuja_orlin_layered,
            **btn_style,
        ).pack(side=tk.LEFT, padx=4)

        tk.Button(
            self.toolbar,
            text="Preflux",
            command=self._on_preflux,
            **btn_style,
        ).pack(side=tk.LEFT, padx=4)

        tk.Button(
            self.toolbar,
            text="Preflux Q",
            command=self._on_preflux_queue,
            **btn_style,
        ).pack(side=tk.LEFT, padx=4)

        tk.Button(
            self.toolbar,
            text="Gabow",
            command=self._on_gabow,
            **btn_style,
        ).pack(side=tk.LEFT, padx=4)

        tk.Button(
            self.toolbar,
            text="Reset (R)",
            command=lambda: self._on_reset_key(None),
            **btn_style,
        ).pack(side=tk.LEFT, padx=4)

        tk.Button(
            self.toolbar,
            text="Save",
            command=lambda: self._on_save_key(None),
            **btn_style,
        ).pack(side=tk.LEFT, padx=4)

        tk.Button(
            self.toolbar,
            text="Load",
            command=lambda: self._on_load_key(None),
            **btn_style,
        ).pack(side=tk.LEFT, padx=4)

        # Status bar – shows algorithm phase / delta label
        self.status_label = tk.Label(
            self.root,
            text="",
            bg="#1A252F",
            fg="#ECF0F1",
            font=("Segoe UI", 9),
            anchor="w",
            padx=10,
        )
        self.status_label.pack(side=tk.TOP, fill=tk.X)

        self.canvas = Canvas(self.root, width=width, height=height, bg="#F5F5F5")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.nodes = {}
        self.edges = {}

        self.dragging_node = None
        self.drag_offset_x = 0
        self.drag_offset_y = 0
        self.drag_started = False

        self.selected_nodes = []

        self.default_color = "#4ECDC4"
        self.selected_color = "#FF6B6B"
        self.edge_color = "#2C3E50"
        self.edge_label_bg = "#FFFBEA"
        self.edge_label_border = "#2C3E50"

        self.canvas.bind("<Button-1>", self._on_mouse_down)
        self.canvas.bind("<B1-Motion>", self._on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_mouse_up)
        self.canvas.bind("<Button-3>", self._on_right_click)

        self.root.bind("<Configure>", self._on_window_resize)
        self.root.bind("<f>", self._on_max_flow_key)
        self.root.bind("<F>", self._on_max_flow_key)
        self.root.bind("<r>", self._on_reset_key)
        self.root.bind("<R>", self._on_reset_key)
        self.root.bind("<Control-s>", self._on_save_key)
        self.root.bind("<Control-o>", self._on_load_key)

    def _on_save_key(self, event):
        """Handle Ctrl+S - save graph to file."""
        from tkinter import filedialog

        file_path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            title="Save Graph",
        )
        if file_path:
            self.controller.save_graph(file_path)

    def _on_load_key(self, event):
        """Handle Ctrl+O - load graph from file."""
        from tkinter import filedialog

        file_path = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            title="Load Graph",
        )
        if file_path:
            self.controller.load_graph(file_path)

    def _on_max_flow_key(self, event):
        """Handle 'f' key - run max flow from first to last node."""
        if not self.controller.node_positions:
            print("No nodes in the graph")
            return

        source = min(self.controller.node_positions.keys())
        sink = max(self.controller.node_positions.keys())

        if source == sink:
            print("Need at least 2 nodes for max flow")
            return

        self._clear_selection()
        self.controller.run_max_flow(source, sink, delay_ms=1500)

    def _on_ford_fulkerson(self):
        """Handle Ford-Fulkerson button click."""
        if not self.controller.node_positions:
            print("No nodes in the graph")
            return

        source = min(self.controller.node_positions.keys())
        sink = max(self.controller.node_positions.keys())

        if source == sink:
            print("Need at least 2 nodes for Ford-Fulkerson")
            return

        self._clear_selection()
        self.controller.run_ford_fulkerson(source, sink, delay_ms=1500)

    def _on_edmonds_karp(self):
        """Handle Edmonds-Karp button click."""
        if not self.controller.node_positions:
            print("No nodes in the graph")
            return

        source = min(self.controller.node_positions.keys())
        sink = max(self.controller.node_positions.keys())

        if source == sink:
            print("Need at least 2 nodes for Edmonds-Karp")
            return

        self._clear_selection()
        self.controller.run_edmonds_karp(source, sink, delay_ms=1500)

    def _on_ahuja_orlin(self):
        """Handle Ahuja-Orlin button click."""
        if not self.controller.node_positions:
            print("No nodes in the graph")
            return

        source = min(self.controller.node_positions.keys())
        sink = max(self.controller.node_positions.keys())

        if source == sink:
            print("Need at least 2 nodes for Ahuja-Orlin")
            return

        self._clear_selection()
        self.controller.run_ahuja_orlin(source, sink, delay_ms=1500)

    def _on_ahuja_orlin_shortest_path(self):
        """Handle Ahuja-Orlin shortest-path button click."""
        if not self.controller.node_positions:
            print("No nodes in the graph")
            return

        source = min(self.controller.node_positions.keys())
        sink = max(self.controller.node_positions.keys())

        if source == sink:
            print("Need at least 2 nodes for Ahuja-Orlin shortest path")
            return

        self._clear_selection()
        self.controller.run_ahuja_orlin_shortest_path(source, sink, delay_ms=1500)

    def _on_ahuja_orlin_layered(self):
        """Handle Ahuja-Orlin layered networks button click."""
        if not self.controller.node_positions:
            print("No nodes in the graph")
            return

        source = min(self.controller.node_positions.keys())
        sink = max(self.controller.node_positions.keys())

        if source == sink:
            print("Need at least 2 nodes for Ahuja-Orlin layered networks")
            return

        self._clear_selection()
        self.controller.run_ahuja_orlin_layered(source, sink, delay_ms=1500)

    def _on_gabow(self):
        """Handle Gabow button click."""
        if not self.controller.node_positions:
            print("No nodes in the graph")
            return

        source = min(self.controller.node_positions.keys())
        sink = max(self.controller.node_positions.keys())

        if source == sink:
            print("Need at least 2 nodes for Gabow")
            return

        self._clear_selection()
        self.controller.run_gabow(source, sink, delay_ms=1500)

    def _on_preflux(self):
        """Handle Preflux button click."""
        if not self.controller.node_positions:
            print("No nodes in the graph")
            return

        source = min(self.controller.node_positions.keys())
        sink = max(self.controller.node_positions.keys())

        if source == sink:
            print("Need at least 2 nodes for Preflux")
            return

        self._clear_selection()
        self.controller.run_preflux(source, sink, delay_ms=1500)

    def _on_preflux_queue(self):
        """Handle Preflux Queue button click."""
        if not self.controller.node_positions:
            print("No nodes in the graph")
            return

        source = min(self.controller.node_positions.keys())
        sink = max(self.controller.node_positions.keys())

        if source == sink:
            print("Need at least 2 nodes for Preflux Queue")
            return

        self._clear_selection()
        self.controller.run_preflux_queue(source, sink, delay_ms=1500)

    def _on_reset_key(self, event):
        """Handle 'r' key - reset all flows to zero and refresh."""
        for edges in self.controller.graph.edges.values():
            for edge in edges:
                edge.flow = 0
        self.controller.reset_colors()
        self.controller.update_edge_labels()
        print("All flows reset to 0")

    def _on_right_click(self, event):
        """Handle right-click - delete node if cursor is over one."""
        x, y = event.x, event.y

        clicked_node_id = self._get_node_at_position(x, y)

        if clicked_node_id is not None:
            self._delete_node(clicked_node_id)

    def _delete_node(self, node_id: int):
        """Delete a node and all its connected edges."""
        if node_id not in self.nodes:
            return

        edges_to_remove = []
        for start, end in self.edges.keys():
            if start == node_id or end == node_id:
                edges_to_remove.append((start, end))

        for edge_key in edges_to_remove:
            start, end = edge_key
            self.controller.graph.remove_edge(start, end)
            self._remove_edge(start, end)

        if node_id in self.controller.node_positions:
            del self.controller.node_positions[node_id]

        if node_id in self.controller.graph.edges:
            del self.controller.graph.edges[node_id]

        self.nodes[node_id].delete()
        del self.nodes[node_id]

        if node_id in self.selected_nodes:
            self.selected_nodes.remove(node_id)

    def _on_window_resize(self, event):
        """Handle window resize event."""
        if event.widget == self.root:
            self.width = self.canvas.winfo_width()
            self.height = self.canvas.winfo_height()

    def _on_mouse_down(self, event):
        """Handle mouse button press - start dragging or select node."""
        x, y = event.x, event.y

        clicked_node_id = self._get_node_at_position(x, y)

        if clicked_node_id is not None:

            self.dragging_node = clicked_node_id
            self.drag_started = False
            node_pos = self.nodes[clicked_node_id].position
            self.drag_offset_x = x - node_pos[0]
            self.drag_offset_y = y - node_pos[1]
        else:

            self._clear_selection()
            self._add_node_at_position(x, y)

    def _on_mouse_drag(self, event):
        """Handle mouse drag - move the node being dragged."""
        if self.dragging_node is not None:
            self.drag_started = True
            x, y = event.x, event.y
            new_x = x - self.drag_offset_x
            new_y = y - self.drag_offset_y

            final_position = self.controller.move_node(
                self.dragging_node, (new_x, new_y)
            )

            if final_position is not None:

                self.nodes[self.dragging_node].update_position(final_position)

                self._update_node_edges(self.dragging_node)

    def _on_mouse_up(self, event):
        """Handle mouse button release - stop dragging or select node for edge creation."""
        if self.dragging_node is not None:
            if not self.drag_started:

                self._toggle_node_selection(self.dragging_node)

            self.dragging_node = None
            self.drag_started = False

    def _toggle_node_selection(self, node_id: int):
        """Toggle node selection for edge creation."""
        if node_id in self.selected_nodes:

            self.selected_nodes.remove(node_id)
            self.nodes[node_id].set_color(self.default_color)
        else:

            if len(self.selected_nodes) >= 2:
                self._clear_selection()

            self.selected_nodes.append(node_id)
            self.nodes[node_id].set_color(self.selected_color)

            if len(self.selected_nodes) == 2:
                start, end = self.selected_nodes
                self._toggle_edge(start, end)
                self._clear_selection()

    def _toggle_edge(self, start: int, end: int):
        """Add or remove edge between two nodes."""

        edge_exists = self._edge_exists(start, end)

        if edge_exists:

            self.controller.graph.remove_edge(start, end)
            self._remove_edge(start, end)
        else:

            dialog = EdgeDialogWidget(self.root, start, end)
            result = dialog.get_result()

            if result is None:
                return

            capacity, flow = result

            self.controller.add_edge(start, end, capacity=capacity, flow=flow)
            self._add_edge(start, end, capacity, flow)

    def _edge_exists(self, start: int, end: int) -> bool:
        """Check if edge exists in graph."""
        if start not in self.controller.graph.edges:
            return False
        return any(edge.end == end for edge in self.controller.graph.edges[start])

    def _clear_selection(self):
        """Clear all selected nodes."""
        for node_id in self.selected_nodes:
            if node_id in self.nodes:
                self.nodes[node_id].set_color(self.default_color)
        self.selected_nodes.clear()

    def _get_node_at_position(self, x: int, y: int) -> int | None:
        """Check if there's a node at the given position."""
        for node_id, node_visual in self.nodes.items():
            if node_visual.contains_point(x, y):
                return node_id
        return None

    def _add_node_at_position(self, x: int, y: int):
        """Add a new node at the specified position."""

        if self.controller.node_positions:
            node_id = max(self.controller.node_positions.keys()) + 1
        else:
            node_id = 1

        final_position = self.controller.add_node((x, y))

        if final_position is None:
            print(f"Warning: Failed to add node at ({x}, {y})")
            return

        node_visual = NodeWidget(
            self.canvas,
            node_id,
            final_position,
            self.controller.node_radius,
            self.default_color,
        )
        self.nodes[node_id] = node_visual

    def _add_edge(self, start: int, end: int, capacity: int, flow: int):
        """Add an edge visual."""
        if start not in self.nodes or end not in self.nodes:
            return

        edge_visual = EdgeWidget(
            self.canvas,
            self.nodes[start],
            self.nodes[end],
            capacity,
            flow,
            self.edge_color,
            self.edge_label_bg,
            self.edge_label_border,
        )
        self.edges[(start, end)] = edge_visual

    def _remove_edge(self, start: int, end: int):
        """Remove an edge visual."""
        if (start, end) in self.edges:
            self.edges[(start, end)].delete()
            del self.edges[(start, end)]

    def _update_node_edges(self, node_id: int):
        """Update all edges connected to a node."""
        for (start, end), edge_visual in self.edges.items():
            if start == node_id or end == node_id:
                edge_visual.update()

    def refresh(self):
        """Redraw all nodes and edges."""
        self.canvas.delete("all")
        self.nodes.clear()
        self.edges.clear()

        for node_id, position in self.controller.node_positions.items():
            node_visual = NodeWidget(
                self.canvas,
                node_id,
                position,
                self.controller.node_radius,
                self.default_color,
            )
            self.nodes[node_id] = node_visual

        try:
            for start_node in self.controller.node_positions.keys():
                if start_node in self.controller.graph.edges:
                    for edge in self.controller.graph.edges[start_node]:
                        if edge.end in self.controller.node_positions:
                            self._add_edge(
                                start_node, edge.end, edge.capacity, edge.flow
                            )
        except AttributeError:
            pass

    def run(self):
        """Start the Tkinter main loop."""
        self.root.mainloop()

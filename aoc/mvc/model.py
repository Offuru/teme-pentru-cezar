class Edge:
    def __init__(
        self,
        start: int,
        end: int,
        capacity: int,
        flow: int = 0,
    ):
        self.start = start
        self.end = end
        self.capacity = capacity
        self.flow = flow


class Graph:
    def __init__(
        self, vertex_count: int = 0, edges: dict[tuple[int, int], tuple[int, int]] = {}
    ):
        self.vertex_count: int = vertex_count
        self.edges: dict[int, list[Edge]] = {}

        for vertex in range(1, vertex_count + 1):
            self.edges[vertex] = []

        for start, end in edges:
            capacity, flow = edges[(start, end)]
            self.add_edge(start, end, capacity, flow)

    def add_edge(self, start: int, end: int, capacity: int = 0, flow: int = 0):
        """Add an edge from start to end with given capacity and flow."""
        # Ensure the start node exists in the edges dictionary
        if start not in self.edges:
            self.edges[start] = []

        # Add the edge
        self.edges[start].append(Edge(start, end, capacity, flow))

    def add_node(self, node_id: int):
        """Add a node to the graph."""
        if node_id not in self.edges:
            self.edges[node_id] = []

    def __str__(self):
        result = f"Graph with {self.vertex_count} vertices:\n"
        for vertex, edges in self.edges.items():
            result += f"  Vertex {vertex}:\n"
            for edge in edges:
                result += f"    Edge to {edge.end} with capacity {edge.capacity} and flow {edge.flow}\n"
        return result

    def save(self, file_path: str):
        with open(file_path, "w") as f:
            f.write(f"{self.vertex_count}\n\n")
            for edges in self.edges.values():
                for edge in edges:
                    f.write(f"{edge.start} {edge.end} {edge.capacity} {edge.flow}\n")

    def remove_edge(self, u: int, v: int):
        if u in self.edges:
            self.edges[u] = [edge for edge in self.edges[u] if edge.end != v]

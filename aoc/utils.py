from mvc.model import Graph


def read_input(file_path: str) -> Graph:
    with open(file_path, "r") as f:
        lines = f.readlines()
    vertex_count = int(lines[0].strip())

    edges = {}

    for line in lines[1:]:
        if line.strip() == "":
            continue
        input = list(map(int, line.strip().split()))

        if len(input) == 3:
            start, end, capacity = input
            edges[(start, end)] = (capacity, 0)
        elif len(input) == 4:
            start, end, capacity, flow = input
            edges[(start, end)] = (capacity, flow)
        else:
            raise ValueError(f"Invalid line in input file: {line}")

    return Graph(vertex_count, edges)

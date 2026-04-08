from mvc.controller import Controller
from mvc.view import GraphView
from mvc.model import Graph


def main():
    graph = Graph()
    controller = Controller(graph, node_radius=20)

    view = GraphView(controller, width=800, height=600)
    view.run()


if __name__ == "__main__":
    main()

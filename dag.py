import networkx as nx
import matplotlib.pyplot as plt


class DAG:
    def __init__(self):
        self.G = nx.DiGraph()

    # Jobs are nodes; adding an edge implicitly adds nodes
    def add_edge(self, u, v, **attrs):
        """
        u, v are Job objects
        """
        self.G.add_edge(u, v, **attrs)
        if not nx.is_directed_acyclic_graph(self.G):
            self.G.remove_edge(u, v)
            raise ValueError(f"Adding edge {u} → {v} creates a cycle")

    def nodes(self):
        return list(self.G.nodes)

    def edges(self):
        return list(self.G.edges)

    def topo_sort(self):
        return list(nx.topological_sort(self.G))

    def predecessors(self, job):
        return list(self.G.predecessors(job))

    def successors(self, job):
        return list(self.G.successors(job))

    # ---------- Visualization ----------
    def draw(self, left_to_right=False, with_levels=False):
        """
        Visualize the DAG using Graphviz.
        Each node is a Job; labels show job.name.
        """
        args = "-Grankdir=LR" if left_to_right else ""
        pos = nx.nx_agraph.graphviz_layout(self.G, prog="dot", args=args)

        # Labels: Job -> "name (importance)"
        labels = {
            job: f"{job.name}\n"
            for job in self.G.nodes
        }

        if with_levels:
            levels = {}
            for job in nx.topological_sort(self.G):
                preds = list(self.G.predecessors(job))
                levels[job] = 0 if not preds else 1 + max(levels[p] for p in preds)
            node_color = [levels[j] for j in self.G.nodes]
            cmap = plt.cm.Blues
        else:
            node_color = "#E6F2FF"
            cmap = None

        nx.draw(
            self.G,
            pos,
            labels=labels,
            node_size=1800,
            node_color=node_color,
            cmap=cmap,
            edge_color="#555555",
            arrowsize=18,
            font_size=12,
        )

        plt.title("Job DAG")
        plt.show()

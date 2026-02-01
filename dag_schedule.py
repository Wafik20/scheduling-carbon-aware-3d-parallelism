import networkx as nx
import matplotlib.pyplot as plt


class DAGSchedule:
    def __init__(self, dags):
        """
        Args:
            dags (list[DAG]): list of independent DAGs (jobs)
        """
        self.dags = dags

        # Track completed jobs per DAG
        self.completed = {dag: set() for dag in dags}

    # ---------------- Core scheduling logic ----------------

    def available(self):
        """
        Return all currently available jobs,
        across all DAGs.

        Returns:
            list[(DAG, Job)]
        """
        available_jobs = []

        for dag in self.dags:
            done = self.completed[dag]

            for job in dag.nodes():
                if job in done:
                    continue

                preds = dag.predecessors(job)
                if all(p in done for p in preds):
                    available_jobs.append((dag, job))

        return available_jobs

    def consume(self, dag, job):
        """
        Consume (execute) a job.

        Args:
            dag (DAG): the DAG the job belongs to
            job (Job): job to execute
        """
        if dag not in self.completed:
            raise ValueError("Unknown DAG")

        if job in self.completed[dag]:
            raise ValueError(f"Job {job.name} already consumed")

        preds = dag.predecessors(job)
        if not all(p in self.completed[dag] for p in preds):
            raise ValueError(f"Job {job.name} is not available")

        self.completed[dag].add(job)

    # ---------------- State inspection (for visualization) ----------------

    def job_states(self):
        """
        Return job execution state.

        Returns:
            dict[(DAG, Job)] -> {"completed", "available", "blocked"}
        """
        states = {}
        available = set(self.available())

        for dag in self.dags:
            done = self.completed[dag]
            for job in dag.nodes():
                key = (dag, job)
                if job in done:
                    states[key] = "completed"
                elif key in available:
                    states[key] = "available"
                else:
                    states[key] = "blocked"

        return states

    # ---------------- Visualization ----------------

    def draw(self, left_to_right=False, spacing=200):
        """
        Visualize the current schedule state over all DAGs.

        Args:
            left_to_right (bool): draw DAGs left → right if True
            spacing (int): horizontal spacing between DAGs
        """
        pos = {}
        colors = {}
        labels = {}
        x_offset = 0

        states = self.job_states()
        color_map = {
            "completed": "#BBBBBB",
            "available": "#7CFC90",
            "blocked": "#E6F2FF",
        }

        for dag in self.dags:
            args = "-Grankdir=LR" if left_to_right else ""
            sub_pos = nx.nx_agraph.graphviz_layout(dag.G, prog="dot", args=args)

            max_x = max(x for x, y in sub_pos.values())

            for job, (x, y) in sub_pos.items():
                pos[job] = (x + x_offset, y)
                colors[job] = color_map[states[(dag, job)]]
                labels[job] = job.name

            x_offset += max_x + spacing

        # Compose graphs purely for drawing
        all_graphs = nx.compose_all([dag.G for dag in self.dags])

        nx.draw(
            all_graphs,
            pos,
            labels=labels,
            node_color=[colors[j] for j in all_graphs.nodes],
            node_size=1600,
            arrowsize=18,
            edge_color="#555555",
            font_size=12,
        )

        plt.title("DAG Schedule State")
        plt.show()

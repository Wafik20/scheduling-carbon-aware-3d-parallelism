import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


class DAGSchedule:
    def __init__(self, dags, num_machines=1):
        """
        Args:
            dags (list[DAG]): list of independent DAGs
            num_machines (int): number of worker machines
        """
        self.dags = dags
        self.time_step = 0

        # Per-DAG job state tracking
        self.inactive = {dag: set(dag.nodes()) for dag in dags}
        self.active = {dag: set() for dag in dags}
        self.completed = {dag: set() for dag in dags}

        self.num_machines = num_machines
        self.free_machines = num_machines

    # ---------------- Core scheduling logic ----------------

    def available(self):
        """
        Jobs whose predecessors are completed and are inactive.
        Returns:
            list[(DAG, Job)]
        """
        available_jobs = []

        for dag in self.dags:
            done = self.completed[dag]
            for job in self.inactive[dag]:
                preds = dag.predecessors(job)
                if all(p in done for p in preds):
                    available_jobs.append((dag, job))

        return available_jobs

    def consume(self, dag, job):
        """
        Start executing a job (non-preemptive),
        if a machine is available.
        """
        if self.free_machines <= 0:
            raise RuntimeError("No free machines available")

        if job not in self.inactive[dag]:
            raise ValueError(f"Job {job.name} is not inactive")

        preds = dag.predecessors(job)
        if not all(p in self.completed[dag] for p in preds):
            raise ValueError(f"Job {job.name} is not available")

        # Move job: inactive → active
        self.inactive[dag].remove(job)
        self.active[dag].add(job)

        job.activate(self.time_step)
        self.free_machines -= 1

    def step(self):
        """
        Advance time by one unit and complete jobs whose time has elapsed.
        """
        self.time_step += 1

        for dag in self.dags:
            finished = set()

            for job in self.active[dag]:
                if job.time_estimate is not None:
                    if self.time_step - job.start_time >= job.time_estimate:
                        finished.add(job)

            for job in finished:
                self.active[dag].remove(job)
                self.completed[dag].add(job)
                job.complete(self.time_step)
                self.free_machines += 1

    # ---------------- State inspection (for visualization) ----------------

    def job_states(self):
        """
        Return job execution state.
        """
        states = {}

        available = set(self.available())

        for dag in self.dags:
            for job in dag.nodes():
                key = (dag, job)
                if job in self.completed[dag]:
                    states[key] = "completed"
                elif job in self.active[dag]:
                    states[key] = "active"
                elif (dag, job) in available:
                    states[key] = "available"
                else:
                    states[key] = "blocked"

        return states

    # ---------------- Visualization ----------------

    def draw(self, left_to_right=False, spacing=200):
        pos = {}
        colors = {}
        labels = {}
        x_offset = 0

        states = self.job_states()
        color_map = {
            "completed": "#BBBBBB",
            "active": "#FFD700",
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

        # -------- Legend --------
        legend_handles = [
            mpatches.Patch(color=color_map["available"], label="Available"),
            mpatches.Patch(color=color_map["active"], label="Active"),
            mpatches.Patch(color=color_map["completed"], label="Completed"),
            mpatches.Patch(color=color_map["blocked"], label="Blocked"),
        ]

        plt.legend(
            handles=legend_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.05),
            ncol=4,
            frameon=False,
        )

        plt.title(
            f"DAG Schedule State (t={self.time_step}) — "
            f"Free machines: {self.free_machines}/{self.num_machines}"
        )
        plt.show()
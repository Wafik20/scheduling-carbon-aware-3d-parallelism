class Job:
    def __init__(self, name, time_estimate=None):
        """
        Args:
            name (str): job identifier
            time_estimate (float): estimated time to complete job
        """
        if time_estimate is not None and not (0.0 <= time_estimate):
            raise ValueError(
                f"time_estimate must be non-negative, got {time_estimate}"
            )

        self.name = name
        self.time_estimate = time_estimate
        self.start_time = None
        self.end_time = None
        

    def __repr__(self):
        return f"Job({self.name}"

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return isinstance(other, Job) and self.name == other.name
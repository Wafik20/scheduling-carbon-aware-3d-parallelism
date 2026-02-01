class Job:
    # Explicit states (simple strings, no Enum yet)
    INACTIVE = "inactive"
    ACTIVE = "active"
    DONE = "done"

    def __init__(self, name, time_estimate=None):
        """
        Args:
            name (str): job identifier
            time_estimate (float | None): estimated runtime (optional)
        """
        if time_estimate is not None and time_estimate < 0:
            raise ValueError(
                f"time_estimate must be non-negative, got {time_estimate}"
            )

        self.name = name
        self.time_estimate = time_estimate

        # Scheduling-related fields
        self.state = Job.INACTIVE
        self.start_time = None
        self.end_time = None

    # ---- State transitions (minimal, explicit) ----

    def activate(self, start_time):
        if self.state != Job.INACTIVE:
            raise RuntimeError(f"Job {self.name} cannot be activated from {self.state}")
        self.state = Job.ACTIVE
        self.start_time = start_time

    def complete(self, end_time):
        if self.state != Job.ACTIVE:
            raise RuntimeError(f"Job {self.name} cannot complete from {self.state}")
        self.state = Job.DONE
        self.end_time = end_time

    # ---- Utilities ----

    def __repr__(self):
        return f"Job({self.name}, state={self.state})"

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return isinstance(other, Job) and self.name == other.name

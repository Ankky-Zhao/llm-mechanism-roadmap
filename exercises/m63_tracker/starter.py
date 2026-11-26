import json, pathlib, uuid

class RunStore:
    """A minimal experiment tracker: one directory per run, one JSON file per directory."""
    def __init__(self, root):
        self.root = pathlib.Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def start_run(self, params):
        # TODO: create root/<run_id>/run.json with the params, status "running" and no metrics
        raise NotImplementedError

    def log_metric(self, run_id, name, value, step):
        # TODO: append [step, value] to the history of this metric
        raise NotImplementedError

    def end_run(self, run_id, status="finished"):
        raise NotImplementedError

    def get_run(self, run_id):
        raise NotImplementedError

    def best_run(self, name, mode="max"):
        # TODO: among finished runs that logged `name`, compare the last logged values
        raise NotImplementedError

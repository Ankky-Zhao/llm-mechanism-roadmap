import json, pathlib, uuid

class RunStore:
    """A minimal experiment tracker: one directory per run, one JSON file per directory."""
    def __init__(self, root):
        self.root = pathlib.Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _read(self, run_id):
        return json.loads((self.root / run_id / "run.json").read_text())

    def _write(self, run):
        d = self.root / run["id"]; d.mkdir(exist_ok=True)
        (d / "run.json").write_text(json.dumps(run))

    def start_run(self, params):
        run = {"id": uuid.uuid4().hex[:8], "params": dict(params), "status": "running", "metrics": {}}
        self._write(run)
        return run["id"]

    def log_metric(self, run_id, name, value, step):
        run = self._read(run_id)
        run["metrics"].setdefault(name, []).append([step, value])
        self._write(run)

    def end_run(self, run_id, status="finished"):
        run = self._read(run_id); run["status"] = status
        self._write(run)

    def get_run(self, run_id):
        return self._read(run_id)

    def best_run(self, name, mode="max"):
        runs = [self._read(d.name) for d in self.root.iterdir() if (d / "run.json").exists()]
        runs = [r for r in runs if r["status"] == "finished" and r["metrics"].get(name)]
        if not runs:
            return None
        pick = max if mode == "max" else min
        return pick(runs, key=lambda r: r["metrics"][name][-1][1])

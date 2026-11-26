import tempfile, os, logging
logging.getLogger("mlflow").setLevel(logging.ERROR); logging.getLogger("alembic").setLevel(logging.ERROR)
configs = [{"lr": 0.1, "epochs": 3}, {"lr": 0.01, "epochs": 3}, {"lr": 0.001, "epochs": 3}]
final = {0.1: 0.71, 0.01: 0.84, 0.001: 0.78}
curve = lambda c: [final[c["lr"]] - 0.1 * (c["epochs"] - 1 - s) for s in range(c["epochs"])]

tmp = tempfile.mkdtemp()
uri = "sqlite:///" + os.path.join(tmp, "mlflow.db")
log_with_mlflow(uri, "lr-sweep", configs, curve)

client = MlflowClient(tracking_uri=uri)
exp = client.get_experiment_by_name("lr-sweep")
runs = client.search_runs([exp.experiment_id])
assert len(runs) == 3 and all(r.info.status == "FINISHED" for r in runs)
history = client.get_metric_history(runs[0].info.run_id, "val_acc")
assert sorted(m.step for m in history) == [0, 1, 2], "one value per step"
assert all(isinstance(v, str) for v in runs[0].data.params.values()), "MLflow stores params as strings"

params, value = best_with_mlflow(uri, "lr-sweep")
assert params == {"lr": 0.01, "epochs": 3} and abs(value - 0.84) < 1e-9, (params, value)

# the hand-written tracker of the previous exercise reaches the same answer from the same runs
store = RunStore(os.path.join(tmp, "runs"))
for c in configs:
    rid = store.start_run(c)
    for s, v in enumerate(curve(c)): store.log_metric(rid, "val_acc", v, s)
    store.end_run(rid)
mine = store.best_run("val_acc")
assert mine["params"] == params and abs(mine["metrics"]["val_acc"][-1][1] - value) < 1e-9
print("ok")

import tempfile, os
root = tempfile.mkdtemp()
store = RunStore(root)
runs = {}
for lr, final in ((0.1, 0.71), (0.01, 0.84), (0.001, 0.78)):
    rid = store.start_run({"lr": lr, "epochs": 3})
    assert store.get_run(rid)["status"] == "running"
    for step in range(3):
        store.log_metric(rid, "val_acc", final - 0.1 * (2 - step), step)
        store.log_metric(rid, "loss", 1.0 / (step + 1), step)
    store.end_run(rid); runs[lr] = rid
assert len(set(runs.values())) == 3 and all(os.path.isfile(os.path.join(root, r, "run.json")) for r in runs.values())

fresh = RunStore(root)                                   # nothing may live in memory only
r = fresh.get_run(runs[0.01])
assert r["id"] == runs[0.01] and r["params"] == {"lr": 0.01, "epochs": 3} and r["status"] == "finished"
assert [s for s, _ in r["metrics"]["val_acc"]] == [0, 1, 2] and abs(r["metrics"]["val_acc"][-1][1] - 0.84) < 1e-9
assert fresh.best_run("val_acc", "max")["params"]["lr"] == 0.01
assert fresh.best_run("val_acc", "min")["params"]["lr"] == 0.1

crashed = fresh.start_run({"lr": 1.0}); fresh.log_metric(crashed, "val_acc", 0.99, 0)
assert fresh.best_run("val_acc")["params"]["lr"] == 0.01, "a run that is still running is not a candidate"
fresh.end_run(crashed, "failed")
assert fresh.best_run("val_acc")["params"]["lr"] == 0.01, "neither is a failed run"
other = fresh.start_run({"lr": 0.5}); fresh.log_metric(other, "f1", 0.9, 0); fresh.end_run(other)
assert fresh.best_run("val_acc")["params"]["lr"] == 0.01 and fresh.best_run("f1")["params"]["lr"] == 0.5
assert fresh.best_run("unknown") is None
print("ok")

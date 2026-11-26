#!/usr/bin/env python3
"""Run every code exercise.

For each exercises/<id>/ the reference solution must pass tests.py, and the
unfinished starter must fail it (otherwise the tests check nothing).
Usage: python scripts/run_tests.py [--skip-network] [id ...]
"""
import contextlib, gc, io, json, pathlib, sys, time, traceback

root = pathlib.Path(__file__).resolve().parent.parent
args = [a for a in sys.argv[1:] if not a.startswith("--")]
skip_net = "--skip-network" in sys.argv


def load(ex, name):
    return (root / "exercises" / ex / name).read_text(encoding="utf-8")


def run(ex, meta, impl):
    ns = {"__name__": "__exercise__"}
    for dep in meta.get("requires", []):
        exec(compile(load(dep, "solution.py"), f"{dep}/solution.py", "exec"), ns)
    try:
        exec(compile(load(ex, impl), f"{ex}/{impl}", "exec"), ns)
        exec(compile(load(ex, "tests.py"), f"{ex}/tests.py", "exec"), ns)
    finally:
        ns.clear()          # drop models loaded by the exercise before the next one starts
        gc.collect()


failed = 0
for d in sorted((root / "exercises").iterdir()):
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    if meta["kind"] != "code" or (args and d.name not in args):
        continue
    if skip_net and meta.get("network"):
        print(f"SKIP {d.name} (needs network)")
        continue
    t = time.time()
    try:
        run(d.name, meta, "solution.py")
        ok = True
    except Exception:
        ok = False
        traceback.print_exc()
    starter_fails = False
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            run(d.name, meta, "starter.py")
    except Exception:
        starter_fails = True
    status = "ok" if ok and starter_fails else "FAIL"
    if not starter_fails:
        print(f"  starter of {d.name} passes its own tests: the tests are too weak")
    failed += status == "FAIL"
    print(f"{status:4} {d.name}  ({time.time() - t:.1f}s)")
sys.exit(1 if failed else 0)

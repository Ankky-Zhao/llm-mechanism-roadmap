import mlflow
from mlflow.tracking import MlflowClient

def log_with_mlflow(uri, experiment, configs, curve):
    """One MLflow run per config: log its params and the val_acc of every step."""
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment)
    for config in configs:
        with mlflow.start_run():
            mlflow.log_params(config)
            for step, value in enumerate(curve(config)):
                mlflow.log_metric("val_acc", value, step=step)

def best_with_mlflow(uri, experiment, metric="val_acc"):
    """Return (params with numeric values, metric value) of the best run of the experiment."""
    client = MlflowClient(tracking_uri=uri)
    exp_id = client.get_experiment_by_name(experiment).experiment_id
    run = client.search_runs([exp_id], order_by=[f"metrics.{metric} DESC"], max_results=1)[0]
    number = lambda s: float(s) if any(ch in s for ch in ".e") else int(s)
    return {k: number(v) for k, v in run.data.params.items()}, run.data.metrics[metric]

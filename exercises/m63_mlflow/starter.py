import mlflow
from mlflow.tracking import MlflowClient

def log_with_mlflow(uri, experiment, configs, curve):
    """One MLflow run per config: log its params and the val_acc of every step."""
    # TODO: set the tracking uri and the experiment, then start one run per config
    raise NotImplementedError

def best_with_mlflow(uri, experiment, metric="val_acc"):
    """Return (params with numeric values, metric value) of the best run of the experiment."""
    # TODO: search the runs of the experiment ordered by the metric, convert the params back to numbers
    raise NotImplementedError

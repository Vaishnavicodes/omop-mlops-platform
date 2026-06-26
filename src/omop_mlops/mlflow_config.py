"""MLflow experiment configuration.

Switch tracking backend (local, Colab, DagsHub, remote) by setting the
MLFLOW_TRACKING_URI environment variable — no code changes required.

    Local (default): unset or empty → ./mlruns
    DagsHub:         https://dagshub.com/<user>/<repo>.mlflow
    Remote server:   http(s)://<host>:<port>

Load a .env file before importing this module, or set the variable in
your shell / Colab ``os.environ`` before calling ``init_experiment``.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Generator

import mlflow
from dotenv import load_dotenv

load_dotenv()

_DEFAULT_TRACKING_URI = "./mlruns"


def get_tracking_uri() -> str:
    """Return the MLflow tracking URI from the environment or the local default.

    Returns
    -------
    str
        Value of ``MLFLOW_TRACKING_URI`` env var, or ``"./mlruns"`` if unset.
    """
    return os.getenv("MLFLOW_TRACKING_URI") or _DEFAULT_TRACKING_URI


def init_experiment(name: str) -> str:
    """Configure MLflow tracking and create/reuse an experiment.

    Sets the tracking URI from the environment, then creates the experiment
    if it does not exist.

    Parameters
    ----------
    name:
        Human-readable experiment name (e.g. ``"ade-detection-v1"``).

    Returns
    -------
    str
        The MLflow experiment ID.
    """
    uri = get_tracking_uri()
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(name)
    experiment = mlflow.get_experiment_by_name(name)
    assert experiment is not None  # just set it above
    return experiment.experiment_id


@contextmanager
def log_run(
    run_name: str,
    params: dict | None = None,
    metrics: dict | None = None,
    artifact_path: str | None = None,
) -> Generator[mlflow.ActiveRun, None, None]:
    """Context manager that wraps an MLflow run with consistent logging.

    Logs *params*, *metrics*, and optionally an *artifact_path* on exit.

    Parameters
    ----------
    run_name:
        Display name for the MLflow run.
    params:
        Hyperparameters / config to log.
    metrics:
        Scalar metrics to log.
    artifact_path:
        Local file or directory path to upload as an MLflow artifact.

    Example
    -------
    >>> with log_run("train-v1", params={"lr": 0.01}, metrics={"auc": 0.87}):
    ...     model.fit(X_train, y_train)
    """
    with mlflow.start_run(run_name=run_name) as run:
        if params:
            mlflow.log_params(params)
        yield run
        if metrics:
            mlflow.log_metrics(metrics)
        if artifact_path:
            mlflow.log_artifact(artifact_path)

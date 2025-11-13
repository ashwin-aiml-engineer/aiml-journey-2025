"""Day 32.03 — MLflow quickstart (fallback prints when mlflow absent)
Run: ~10 minutes

This demo attempts to import mlflow and log a simple run. If mlflow is not
installed, it prints the equivalent steps so you can try locally after
`pip install mlflow`.
"""
from __future__ import annotations

import time

try:
    import mlflow
    has_mlflow = True
except ImportError:
    has_mlflow = False


def run_demo():
    if not has_mlflow:
        print("mlflow not installed. To try this demo locally:")
        print("pip install mlflow")
        print(
            "Then run this script; it will log a simple run to the local"
        )
        print("MLflow server.")
        print("Example steps:")
        print("- mlflow.set_experiment('day32-demo')")
        print("- with mlflow.start_run():")
        print("    mlflow.log_param('lr', 0.01)")
        print("    mlflow.log_metric('loss', 0.42)")
        return
    mlflow.set_experiment("day32-demo")
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        mlflow.log_param("lr", 0.01)
        mlflow.log_param("epochs", 3)
        # simulate metrics
        mlflow.log_metric("train_loss", 0.42)
        mlflow.log_metric("val_loss", 0.48)
        # log a tiny artifact
        with open("sample_artifact.txt", "w", encoding="utf-8") as f:
            f.write("sample artifact for day32")
        mlflow.log_artifact("sample_artifact.txt")
        print("Logged run:", run_id)


if __name__ == "__main__":
    start = time.time()
    run_demo()
    print("Done. Duration:", round(time.time() - start, 2), "s")

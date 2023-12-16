import os
import joblib
import mlflow
import argparse
from mlflow.tracking import MlflowClient
from pprint import pprint
from src.get_data import read_params


def log_production_model(config_path):
    """
    Log the production model to mlflow
    """
    config = read_params(config_path)

    # MLFlow configuration
    mlflow_config = config["mlflow_config"]
    remote_server_uri = mlflow_config["remote_server_uri"]
    model_name = mlflow_config["registered_model_name"]
    mlflow.set_tracking_uri(remote_server_uri)

    runs = mlflow.search_runs(experiment_ids="1")
    lowest = runs["metrics.Accuracy"].sort_values(ascending=True)[0]
    lowest_run_id = runs[runs["metrics.Accuracy"] == lowest]["run_id"][0]
    logged_model = ""

    client = MlflowClient()
    for mv in client.search_model_versions(f"name = '{model_name}'"):
        mv = dict(mv)

        if mv["run_id"] == lowest_run_id:
            current_version = mv["version"]
            logged_model = mv["source"]
            pprint(mv, indent=4)
            status = client.transition_model_version_stage(name=model_name, version=current_version, stage="Production")
            print(current_version)
            print(logged_model)

        else:
            current_version = mv["version"]
            client.transition_model_version_stage(name=model_name, version=current_version, stage="Staging")
            print(current_version)

    loaded_model = mlflow.pyfunc.load_model(logged_model)
    model_path = os.path.join(os.path.split(os.path.dirname(__file__))[0], config["webapp_model_dir"])

    joblib.dump(loaded_model, model_path)


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--config', type=str, default="params.yaml")
    parsed_args = args.parse_args()
    data = log_production_model(config_path=parsed_args.config)
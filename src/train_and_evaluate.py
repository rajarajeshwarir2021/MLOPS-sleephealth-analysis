import os
import argparse
import mlflow
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from urllib.parse import urlparse
from get_data import read_params


def train_and_evaulate(config_path):
    """
    Load the train and test data from the data/processed folder, Train the model on the train data, and Save the model metrics and parameters
    """
    config = read_params(config_path)

    random_state = config["base"]["random_state"]
    target = config["base"]["target_col"]

    train_data_path = config["split_data"]["train_path"]
    test_data_path = config["split_data"]["test_path"]
    model_path = config["model_dir"]

    p_criterion = config["estimators"]["DecisionTreeClassifier"]["params"]["criterion"]
    p_max_depth = config["estimators"]["DecisionTreeClassifier"]["params"]["max_depth"]

    df_train = pd.read_csv(train_data_path, sep=",", encoding="utf-8")
    df_test = pd.read_csv(test_data_path, sep=",", encoding="utf-8")

    # Formulate the train dataset
    X_train = df_train.drop(columns=[target], axis=1)
    y_train = df_train[target]

    # Formulate the test dataset
    X_test = df_test.drop(columns=[target], axis=1)
    y_test = df_test[target]

    # MLFlow configuration
    mlflow_config = config["mlflow_config"]
    remote_server_uri = mlflow_config["remote_server_uri"]
    mlflow.set_tracking_uri(remote_server_uri)
    mlflow.set_experiment(mlflow_config["experiment_name"])

    with mlflow.start_run(run_name=mlflow_config["run_name"]) as mlops_run:

        # Train and fit the model
        model = DecisionTreeClassifier(criterion=p_criterion, max_depth=p_max_depth, random_state=random_state)
        model.fit(X_train, y_train)

        # Predict on the test data
        y_pred = model.predict(X_test)

        # Evaluate and print the metrics
        accuracy = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)

        # Log parameters and metrics
        mlflow.log_param("criterion", p_criterion)
        mlflow.log_param("max_depth", p_max_depth)

        mlflow.log_metric("Accuracy", accuracy)
        mlflow.log_metric("true_positive", conf_matrix[0][0])
        mlflow.log_metric("true_negative", conf_matrix[1][1])
        mlflow.log_metric("false_positive", conf_matrix[0][1])
        mlflow.log_metric("false_negative", conf_matrix[1][0])

        # Log the confusion matrix
        fig = plt.figure()
        plt.matshow(conf_matrix)
        plt.title('Confusion Matrix: Sleep Health')
        plt.colorbar()
        plt.ylabel('True Label')
        plt.xlabel('Predicated Label')
        plt.savefig('confusion_matrix.png')
        plt.close(fig)
        mlflow.log_artifact('confusion_matrix.png')

        # Save the model
        tracking_url_type_store = urlparse(mlflow.get_artifact_uri()).scheme
        if tracking_url_type_store != "file":
            mlflow.sklearn.log_model(model, "model", registered_model_name=mlflow_config["registered_model_name"])
        else:
            mlflow.sklearn.log_model(model, "model", registered_model_name=mlflow_config["registered_model_name"])


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--config', type=str, default="params.yaml")
    parsed_args = args.parse_args()
    train_and_evaulate(config_path=parsed_args.config)
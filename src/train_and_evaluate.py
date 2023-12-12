import os
import argparse
import json
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
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

    # Train and fit the model
    model = DecisionTreeClassifier(criterion=p_criterion, max_depth=p_max_depth, random_state=random_state)
    model.fit(X_train, y_train)

    # Predict on the test data
    y_pred = model.predict(X_test)

    # Evaluate and print the metrics
    accuracy = accuracy_score(y_test, y_pred)
    con_matrix = confusion_matrix(y_test, y_pred)
    print(f"DecisionTreeClassifier model (criterion={p_criterion}, max_depth={p_max_depth}) \nAccuracy: {round(accuracy*100, 2)}%")

    # Generate report files
    scores_file = config["reports"]["scores"]
    params_file = config["reports"]["params"]
    confusion_matrix_file = config["reports"]["figures"]

    with open(scores_file, "w") as f:
        scores = {
            "Accuracy": accuracy,
        }
        json.dump(scores, f, indent=4)

    with open(params_file, "w") as f:
        params = {
            "criterion": p_criterion,
            "max_depth": p_max_depth,
        }
        json.dump(params, f, indent=4)

    fig = plt.figure()
    plt.matshow(con_matrix)
    plt.title('Confusion Matrix: Sleep Health')
    plt.colorbar()
    plt.ylabel('True Label')
    plt.xlabel('Predicated Label')
    plt.savefig(os.path.join(confusion_matrix_file, 'confusion_matrix.jpg'))

    # Save the model
    os.makedirs(model_path, exist_ok=True)
    model_file_path = os.path.join(model_path, "model.joblib")
    joblib.dump(model, model_file_path)


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--config', type=str, default="params.yaml")
    parsed_args = args.parse_args()
    train_and_evaulate(config_path=parsed_args.config)
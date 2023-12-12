import os
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from get_data import read_params


def split_and_save(config_path):
    """
    Split the raw data into train and test sets, and save the train and test sets to the data/processed folder for further processing
    """
    config = read_params(config_path)
    random_state = config["base"]["random_state"]
    raw_data_path = config["load_data"]["raw_dataset_csv"]
    train_data_path = config["split_data"]["train_path"]
    test_data_path = config["split_data"]["test_path"]
    split_ratio = config["split_data"]["test_size"]

    df = pd.read_csv(raw_data_path, sep=",", encoding="utf-8")
    train, test = train_test_split(df, test_size=split_ratio, random_state=random_state)
    train.to_csv(train_data_path, sep=",", encoding="utf-8", index=False)
    test.to_csv(test_data_path, sep=",", encoding="utf-8", index=False)


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--config', type=str, default="params.yaml")
    parsed_args = args.parse_args()
    split_and_save(config_path=parsed_args.config)
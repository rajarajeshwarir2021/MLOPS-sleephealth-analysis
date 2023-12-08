import os
import yaml
import pandas as pd
import argparse

def read_params(config_path):
    """
    Read the parameters
    """
    with open(config_path, 'r') as f:
        params = yaml.safe_load(f)
    return params


def get_data(config_path):
    """
    Process the parameters
    """
    config = read_params(config_path)
    data_path = config['data_source']['data_source_path']
    df = pd.read_csv(data_path, sep=",", encoding="utf-8")
    print(df.head())
    return df


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--config', type=str, default="params.yaml")
    parsed_args = args.parse_args()
    data = get_data(config_path=parsed_args.config)
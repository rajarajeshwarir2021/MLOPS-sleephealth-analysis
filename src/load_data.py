import os
import yaml
import json
import joblib
import argparse
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from get_data import read_params, get_data

LABEL_MAP = {}


def label_encode_dataframe(config, df):
    """
    Label encode the categorical columns
    """
    label_encoder = LabelEncoder()
    categorical_cols = ['Gender', 'Occupation', 'BMI_Category', 'Sleep_Disorder']

    for col in categorical_cols:
        df[col] = label_encoder.fit_transform(df[col])

        # Save the Label encoder
        encoder_path = os.path.join(config["load_data"]["label_encoded_data"], f"{col}_LabelEncoder.joblib")
        joblib.dump(label_encoder, encoder_path, compress=9)

        # Get the label mapping
        encoder_mapping = dict(zip(label_encoder.classes_.tolist(), label_encoder.transform(label_encoder.classes_).tolist()))
        LABEL_MAP[col] = encoder_mapping

    print(f"Label Map: {LABEL_MAP}")

    # Save the label mapping
    label_map_path = os.path.join(config["load_data"]["label_encoded_data"], "label_map.json")
    with open(label_map_path, 'w') as f:
        json.dump(LABEL_MAP, f)

    return df


def process_dataframe(df):
    """
    Process the dataframe
    """
    # STEP 1: Drop the Personal ID, Quality of Sleep, and Physical Activity Level column
    df_new = df.drop(['Person ID', 'Quality of Sleep', 'Physical Activity Level'], axis=1)

    # STEP 2: Change Blood Pressure column from categorical to numerical
    df_new = pd.concat([df_new, df_new['Blood Pressure'].str.split('/', expand=True)], axis=1).drop('Blood Pressure', axis=1)
    # Rename the column and convert the datatype from object to int64
    df_new = df_new.rename(columns={0: 'Blood Pressure High', 1: 'Blood Pressure Low'})
    df_new[['Blood Pressure High', 'Blood Pressure Low']] = df_new[['Blood Pressure High', 'Blood Pressure Low']].apply(pd.to_numeric)

    # STEP 3: Replace NaN with None in Sleep Disorder Target column
    df_new.loc[df_new['Sleep Disorder'].isna(), 'Sleep Disorder'] = "Normal"

    # STEP 4: Move the target column to the end
    df_new.insert(len(df_new.columns) - 1, 'Sleep Disorder', df_new.pop('Sleep Disorder'))

    # STEP 5: Rename column names
    new_cols = [col.replace(" ", "_") for col in df_new.columns]
    df_new.columns = new_cols

    return df_new


def load_and_save(config_path):
    """
    Read the data from the datasource and save it to the data/raw folder for further processing
    """
    # Read the parameters
    config = read_params(config_path)

    # Get the dataframe
    df = get_data(config_path)

    # Process the dataframe
    df_processed = process_dataframe(df)

    # Label encode the dataframe
    df_encoded = label_encode_dataframe(config, df_processed)

    # Save the processed dataframe
    raw_data_path = config["load_data"]["raw_dataset_csv"]
    df_encoded.to_csv(raw_data_path, sep=",", encoding="utf-8", index=False)

    print(f"Dataframe: {df_encoded}")
    print(f"Columns: {df_encoded.columns}")


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--config', type=str, default="params.yaml")
    parsed_args = args.parse_args()
    load_and_save(config_path=parsed_args.config)
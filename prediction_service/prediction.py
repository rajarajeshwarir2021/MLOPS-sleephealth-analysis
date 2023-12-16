import os
import json
import yaml
import joblib
import numpy as np
import pandas as pd

params_path = "params.yaml"
schema_path = os.path.join("prediction_service", "schema_data.json")


class NotInRange(Exception):
    """
    Raised when values entered are not in range
    """
    def __init__(self, message="Values entered are not in range"):
        self.message = message
        super().__init__(self.message)


class NotInFeatureColumn(Exception):
    """
    Raised when values entered are not in feature columns
    """
    def __init__(self, message="Values entered are not in feature columns"):
        self.message = message
        super().__init__(self.message)


def read_params(config_path=params_path):
    """
    Read the parameters
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def get_schema(schema_path=schema_path):
    """
    Get the schema
    """
    with open(schema_path, 'r') as f:
        schema = json.load(f)
    return schema


def validate_input(dict_request):
    """
    Validate the input
    """
    def _validate_cols(col):
        schema = get_schema()
        actual_cols = schema.keys()
        if col not in actual_cols:
            raise NotInFeatureColumn

    def _validate_values(col, val):
        schema = get_schema()
        if col in ["Gender", "Occupation", "BMI_Category"]:
            if not val in schema[col].values():
                raise NotInRange
        elif not (schema[col]["min"] <= float(dict_request[col]) <= schema[col]["max"]):
            raise NotInRange

    for col, val in dict_request.items():
        _validate_cols(col)
        _validate_values(col, val)

    return True


def create_dataframe(data):
    """
    Create a data frame from the data
    """
    data = np.array([list(data.values())])
    headers = ['Gender', 'Age', 'Occupation', 'Sleep_Duration', 'Stress_Level', 'BMI_Category', 'Heart_Rate',
               'Daily_Steps', 'Blood_Pressure_High', 'Blood_Pressure_Low']
    df = pd.DataFrame(data, columns=headers)

    return df


def get_label_encoder(item):
    """
    Get the label encoder
    """
    config = read_params(params_path)
    encoder_path = config["label_encoder"][item.lower()]
    # Load the encoder
    encoder = joblib.load(encoder_path)

    return encoder


def encode_label(data):
    """
    Encode the label data
    """
    df = create_dataframe(data)

    for item in ["Gender", "Occupation", "BMI_Category"]:
        label_encoder = get_label_encoder(item)
        df[item] = label_encoder.transform(df[item])

    df = df.values.tolist()[0]

    return df

def decode_label(data):
    """
    Decode the label data
    """
    label_encoder = get_label_encoder("sleep_disorder")
    label = label_encoder.inverse_transform(data)[0]

    return label


def form_response(dict_request):
    """
    Predict the target for the given parameters in dictionary format
    """
    if validate_input(dict_request):
        data_encoded = encode_label(dict_request)
        data = [list(map(float, data_encoded))]
        encoded_response = predict(data)
        response = decode_label(encoded_response)
        return response


def api_response(dict_request):
    """
    Predict the target for the given parameters in json format
    """
    try:
        if validate_input(dict_request):
            data_encoded = encode_label(dict_request)
            data = [list(map(float, data_encoded))]
            encoded_response = predict(data=data)
            response = decode_label(encoded_response)
            response = {"response": response}
            return response
    except NotInRange as e:
        response = {"the_exected_range": get_schema(), "response": str(e) }
        return response

    except NotInFeatureColumn as e:
        response = {"the_exected_cols": get_schema().keys(), "response": str(e) }
        return response

    except Exception as e:
        response = {"Expected range": get_schema(), "response": str(e)}
        return response


def predict(data):
    """
    Predict the target for the given parameters in list format
    """
    config = read_params(params_path)
    model_dir_path = config['webapp_model_dir']
    model = joblib.load(model_dir_path)
    prediction = [model.predict(data)][0]
    try:
        if 0 <= prediction <= 2:
            return prediction
        else:
            raise NotInRange
    except NotInRange:
        return "Unexpected result"



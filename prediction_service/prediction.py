import os
import json
import yaml
import joblib
import numpy as np

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
        if not (schema[col]["min"] <= float(dict_request[col]) <= schema[col]["max"]):
            raise NotInRange

    for col, val in dict_request.items():
        _validate_cols(col)
        _validate_values(col, val)

    return True


def form_response(dict_request):
    """
    Predict the target for the given parameters in dictionary format
    """
    if validate_input(dict_request):
        data = dict_request.values()
        data_array = [data]
        data_encoded = encode_label(data)
        data = [list(map(float, data_encoded))]
        response = predict(data)
        response = decode_label(response)
        return response


def api_response(dict_request):
    """
    Predict the target for the given parameters in json format
    """
    try:
        if validate_input(dict_request):
            data = np.array([list(dict_request.values())])
            response = predict(data=data)
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
    prediction = [model.predict(data)][0][0]
    print(prediction)
    try:
        if 0 <= prediction <= 10:
            return prediction
        else:
            raise NotInRange
    except NotInRange:
        return "Unexpected result"



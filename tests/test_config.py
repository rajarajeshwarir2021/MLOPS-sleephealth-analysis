import pytest
import prediction_service
from prediction_service.prediction import form_response, api_response

input_data = {
    "incorrect_range":
    {
        "Gender": "Others",
        "Age": 18,
        "Occupation": "Pilot",
        "Sleep_Duration": 10,
        "Stress_Level": 10,
        "BMI_Category": "Underweight",
        "Heart_Rate": 90,
        "Daily_Steps": 2000,
        "Blood_Pressure_High": 150,
        "Blood_Pressure_Low": 90
    },

    "correct_range":
    {
        "Gender": "Male",
        "Age": 30,
        "Occupation": "Software Engineer",
        "Sleep_Duration": 6,
        "Stress_Level": 6,
        "BMI_Category": "Normal",
        "Heart_Rate": 70,
        "Daily_Steps": 5000,
        "Blood_Pressure_High": 120,
        "Blood_Pressure_Low": 75
    },

    "incorrect_column":
    {
        "Gender": "Male",
        "Age": 30,
        "Occupation": "Software Engineer",
        "Sleep Duration": 6,
        "Stress Level": 6,
        "BMI Category": "Normal",
        "HeartRate": 70,
        "Daily Steps": 5000,
        "Blood Pressure High": 120,
        "Blood_Pressure Low": 75
    }
}

TARGET_VALUES = ["Normal", "Sleep Apnea", "Insomnia"]
TARGET_RESPONSE = {'response': 'Normal'}


def test_form_response_correct_range(data=input_data["correct_range"]):
    res = form_response(data)
    assert  res in TARGET_VALUES


def test_api_response_correct_range(data=input_data["correct_range"]):
    res = api_response(data)
    assert res == TARGET_RESPONSE


def test_form_response_incorrect_range(data=input_data["incorrect_range"]):
    with pytest.raises(prediction_service.prediction.NotInRange):
        res = form_response(data)


def test_api_response_incorrect_range(data=input_data["incorrect_range"]):
    res = api_response(data)
    assert res["response"] == prediction_service.prediction.NotInRange().message


def test_api_response_incorrect_col(data=input_data["incorrect_column"]):
    res = api_response(data)
    assert res["response"] == prediction_service.prediction.NotInFeatureColumn().message
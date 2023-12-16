import json
import os

import numpy as np
import pandas as pd

# schema_path = os.path.join("prediction_service", "schema_data.json")
# with open(schema_path, 'r') as f:
#     schema = json.load(f)
#     print(schema)

dict_request = {'Age': '27', 'BMI_Category': 'Normal', 'Blood_Pressure_High': '115', 'Blood_Pressure_Low': '75', 'Daily_Steps': '3000', 'Gender': 'Male', 'Heart_Rate': '65', 'Occupation': 'Software Engineer', 'Sleep_Duration': '5.8', 'Stress_Level': '6'}
data = np.array([list(dict_request.values())])
print(data)

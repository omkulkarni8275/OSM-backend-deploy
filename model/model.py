import joblib
import numpy as np
import pandas as pd
from model.utils.preprocessing import preprocess_input_data

# Load the saved models and label encoder
ocd_percentage_model = joblib.load('model/models/ocd_percentage_model.joblib')
ocd_severity_model = joblib.load('model/models/ocd_severity_model.joblib')
ocd_severity_label_encoder = joblib.load('model/models/ocd_severity_label_encoder.joblib')

def predict_ocd(data):
    """
    Given a dictionary `data`, this function preprocesses the data,
    and then predicts OCD severity and OCD percentage.
    """
    # Preprocess the input data to match the model input format
    preprocessed_data = preprocess_input_data(data)
    
    # Predict the OCD percentage (regression model)
    predicted_percentage = ocd_percentage_model.predict(preprocessed_data)[0]
    
    # Predict the OCD severity (classification model)
    predicted_severity_encoded = ocd_severity_model.predict(preprocessed_data)[0]
    
    # Decode the severity label
    predicted_severity = ocd_severity_label_encoder.inverse_transform([predicted_severity_encoded])[0]
    
    return predicted_severity, predicted_percentage

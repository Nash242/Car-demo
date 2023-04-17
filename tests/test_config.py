

import json
import logging
import os
import sklearn
import pytest
from prediction_service.prediction import form_response, api_response
import prediction_service

input_data = {
    "incorrect_range": 
    {"Year": 2024, 
    "Present_Price": 0, 
    "Kms_Driven": 0, 
    "Fuel_Type": 3, 
    "Seller_Type": 2, 
    "Transmission": 2, 
    "Owner": 3, 
    },

    "correct_range":
    {"Year": 2016, 
    "Present_Price": 8.6, 
    "Kms_Driven": 15000, 
    "Fuel_Type": 1, 
    "Seller_Type": 0, 
    "Transmission": 1, 
    "Owner": 0,
    },

    "incorrect_col":
    {"Year": 2024, 
    "Present_Price": 0, 
    "Kms_Driven": 0, 
    "Fuel_Type": 3, 
    "Seller_Type": 2, 
    "Transmission": 2, 
    "Owner": 3,
    }
}

TARGET_range = {
    "min": 0.1,
    "max": 35.0
}

def test_form_response_correct_range(data=input_data["correct_range"]):
    res = form_response(data)
    assert  TARGET_range["min"] <= res <= TARGET_range["max"]

def test_api_response_correct_range(data=input_data["correct_range"]):
    res = api_response(data)
    assert  TARGET_range["min"] <= res["response"] <= TARGET_range["max"]

def test_form_response_incorrect_range(data=input_data["incorrect_range"]):
    with pytest.raises(prediction_service.prediction.NotInRange):
        res = form_response(data)

def test_api_response_incorrect_range(data=input_data["incorrect_range"]):
    res = api_response(data)
    assert res["response"] == prediction_service.prediction.NotInRange().message

def test_api_response_incorrect_col(data=input_data["incorrect_col"]):
    res = api_response(data)
    assert res["response"] == prediction_service.prediction.NotInCols().message


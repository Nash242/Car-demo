# cleaning or preprocessing
import pandas as pd
import numpy as np
import argparse
from get_data import get_data, read_params

def clean_data(config_path):
    config = read_params(config_path)
    df = get_data(config_path)
    df2 = df.replace({'Fuel_Type':{'Petrol':0, 'Diesel':1, 'CNG':2}})
    df3 = df2.replace({'Seller_Type':{'Dealer':0, 'Individual':1}})
    df4 = df3.replace({'Transmission':{'Manual':0, 'Automatic':1}})
    df1 = df4.drop(['Car_Name'], axis=1)
    # print(df1.head())
    # print(df4.head())
    clean_data_path = config["clean_data"]["clean_dataset_csv"]
    df1.to_csv(clean_data_path, sep=",", index=False)

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    clean_data(config_path=parsed_args.config)
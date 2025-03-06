import pandas as pd
import numpy as np
import logging


## Logging Configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(path):
    """Loads data from a CSV file"""
    try:
        logging.info("Loading data from CSV file >>> ")
        df = pd.read_csv(path, low_memory=False)
        logging.info("Data Loaded Successfully!!")
        return df
    except Exception as e:
        logging.error(f"An Error Occurred While Loading: {e}")
        return None  # Ensure function always returns a value

def change_datatypes(df):
    """ Changes data types of columns to their appropriate format"""
    logging.info("Changing Data Types to appropriate format >>> ")

    df['Date'] = pd.to_datetime(df['Date'])

    # Fill NaN before converting to integers
    df['CompetitionOpenSinceYear'] = df['CompetitionOpenSinceYear'].fillna(1900).astype(int)
    df['Promo2SinceYear'] = df['Promo2SinceYear'].fillna(1900).astype(int)

    df['CompetitionOpenSinceMonth'] = df['CompetitionOpenSinceMonth'].fillna(1).astype(int)
    df['Promo2SinceYear'] = df['Promo2SinceYear'].astype(int)

    logging.info("Data types Changed Successfully")


def replace_missing_values(df):
    """ Handle Missing Values """
    logging.info("Handling Missing Values >>> ")
    df['Promo2SinceWeek'] = np.where(df['Promo2']== 0, 0, df['Promo2SinceWeek'])
    df['Promo2SinceYear'] = np.where(df['Promo2']== 0, 1900, df['Promo2SinceYear'])
    df['PromoInterval'] = np.where(df['Promo2']== 0, 'UNKNOWN',df['PromoInterval'])

    df['CompetitionDistance'] = df['CompetitionDistance'].fillna(0)
    df['CompetitionOpenSinceMonth'] = df['CompetitionOpenSinceMonth'].fillna(0)
    df['CompetitionOpenSinceYear'] = df['CompetitionOpenSinceYear'].fillna(1900)
    
    df['Open'] =df['Open'].fillna(df['Open'].mode()[0])

    logging.info("Missing Value Handled!!")



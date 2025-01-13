from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load the machine learning model
model = joblib.load('model_xgb.pkl')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            # Get form data
            store = int(request.form['Store'])
            day_of_week = int(request.form['DayOfWeek'])
            date_str = request.form['Date']
            date = datetime.strptime(date_str, '%Y-%m-%d')
            open_store = int(request.form['Open'])
            promo = int(request.form['Promo'])
            state_holiday = request.form['StateHoliday']
            school_holiday = int(request.form['SchoolHoliday'])
            store_type = request.form['StoreType']
            assortment = request.form['Assortment']
            competition_distance = float(request.form['CompetitionDistance'])
            competition_open_since_month = int(request.form['CompetitionOpenSinceMonth'])
            competition_open_since_year = int(request.form['CompetitionOpenSinceYear'])
            promo2 = int(request.form['Promo2'])
            promo2_since_week = int(request.form['Promo2SinceWeek'])
            promo2_since_year = int(request.form['Promo2SinceYear'])

            # Create a DataFrame with the input features
            input_features = pd.DataFrame([{
                'Store': store,
                'DayOfWeek': day_of_week,
                'Date': date,
                'Open': open_store,
                'Promo': promo,
                'StateHoliday': state_holiday,
                'SchoolHoliday': school_holiday,
                'StoreType': store_type,
                'Assortment': assortment,
                'CompetitionDistance': competition_distance,
                'CompetitionOpenSinceMonth': competition_open_since_month,
                'CompetitionOpenSinceYear': competition_open_since_year,
                'Promo2': promo2,
                'Promo2SinceWeek': promo2_since_week,
                'Promo2SinceYear': promo2_since_year,
            }])

            # Generate additional features (replicate training feature engineering)
            input_features['Year'] = input_features['Date'].dt.year
            input_features['Month'] = input_features['Date'].dt.month
            input_features['Day'] = input_features['Date'].dt.day
            input_features['Quarter'] = input_features['Date'].dt.quarter

            input_features['IsBeginningOfMonth'] = input_features['Day'].apply(lambda x: 1 if x <= 7 else 0)
            input_features['IsMidMonth'] = input_features['Day'].apply(lambda x: 1 if 8 <= x <= 21 else 0)
            input_features['IsEndOfMonth'] = input_features['Day'].apply(lambda x: 1 if x > 21 else 0)

            # From Day of Week
            input_features['IsWeekend'] = input_features['DayOfWeek'].isin([1, 5]).astype(int)
            input_features['IsWeekday'] = input_features['DayOfWeek'].isin([1, 5]).astype(int)
            
            # Sales and Customers (since they are set to 0, we don't need to calculate these)
            input_features['SalesPerCustomer'] = 0
            input_features['LogSales'] = 0
            

            # Lag features (use dummy values for demonstration; in production, use historical data)
            input_features['SalesLag7'] = 0  # Replace with actual lagged data
            input_features['SalesRollingMean'] = 0  # Replace with actual rolling mean data

            required_order = [
                                'Store', 'StoreType', 'Assortment', 'CompetitionDistance', 
                                'CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear', 'Promo2', 
                                'Promo2SinceWeek', 'Promo2SinceYear', 'DayOfWeek', 'Date', 
                                'Open', 'Promo', 'StateHoliday', 'SchoolHoliday', 
                                'Year', 'Month','Day', 'Quarter', 'IsBeginningOfMonth', 
                                'IsMidMonth', 'IsEndOfMonth','IsWeekday', 'IsWeekend', 'SalesPerCustomer', 
                                'LogSales', 'SalesLag7', 'SalesRollingMean']
            
            for col in required_order:
                if col not in input_features.columns:
                    input_features[col] = 0 

            #reorder
            input_features = input_features[required_order]

            # Drop unused columns (like 'Date')
            input_features.drop(columns=['Date'], inplace=True)

            ## Encode categorical Variables
            non_numeric_columns = input_features.select_dtypes(include=['object']).columns
            label_encoder = LabelEncoder()
            for col in non_numeric_columns:
                input_features[col] = label_encoder.fit_transform(input_features[col].astype(str))

            ## Scale the data
            columns_to_be_scaled = ['CompetitionDistance', 'SalesPerCustomer', 'LogSales', 'SalesLag7', 'SalesRollingMean']

            scaler = StandardScaler()
            input_features[columns_to_be_scaled] = scaler.fit_transform(input_features[columns_to_be_scaled])

            # Make prediction
            prediction = model.predict(input_features)[0]
           

            # Render result
            return render_template('result.html', store=store, prediction=prediction, date=date, day_of_week=day_of_week,
                               open_store=open_store, promo=promo, state_holiday=state_holiday,
                               school_holiday=school_holiday, store_type=store_type, assortment=assortment,
                               competition_distance=competition_distance, competition_open_since_month=competition_open_since_month,
                               competition_open_since_year=competition_open_since_year, promo2=promo2,
                               promo2_since_week=promo2_since_week, promo2_since_year=promo2_since_year)

        except Exception as e:
            return render_template('error.html', error_message=str(e))

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
           
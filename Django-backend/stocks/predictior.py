import re
import json
import csv
from io import StringIO
from bs4 import BeautifulSoup
import requests
import os
import math
import pandas_datareader as web
import numpy as np
import pandas as pd
from pandas import Timestamp
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from mpl_finance import candlestick_ohlc
import matplotlib.dates as mpl_dates
from datetime import timedelta, datetime


def get_stock_quote(stock, scaler, model):
    # # get stock quote
    stock_quote = web.DataReader(stock, data_source='yahoo',
                        start='2015-12-20', end='2020-12-20')

    new_df = stock_quote.filter(['Close'])

    # Get the last 60 day closing price
    last_60_days = new_df[-60:].values

    # Scaling between 0 and 1
    last_60_days_scaled = scaler.transform(last_60_days)

    x_test = []

    # Append the past 60 days
    x_test.append(last_60_days_scaled)

    # Convert to numpy array
    x_test = np.array(x_test)

    # Reshape the data
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    # get the predicted price
    pred_price = model.predict(x_test)

    #undo the scaling
    pred_price = scaler.inverse_transform(pred_price)

    print(pred_price)

    act_price = web.DataReader(stock, data_source='yahoo',
                        start='2020-12-21', end='2020-12-21')

    print(act_price['Close'])


def get_past_predictions(dataset, scaled_data, training_data_len, model, scaler):

    #Create the testing dataset
    test_data = scaled_data[training_data_len-60:,:]
    x_test = []
    y_test= dataset[training_data_len:,:] 
    for i in range(60, len(test_data)):
        x_test.append(test_data[i-60:i, 0])


    # Convert the data to a numpy array
    x_test = np.array(x_test)

    # Reshape the data
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    # Get the models predicted price values(Unscaling predictions)
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)


    # Get the root mean squared error (RMSE)
    rmse = np.sqrt(np.mean(predictions - y_test)**2)

    return predictions


def get_future_predictions(dataset, scaled_data, training_data_len, model, scaler, future_period):
    # Create the future testing dataset
    test_data = scaled_data[training_data_len-60:,:]


    # print(test_data)
    x_test = []
    test_len = len(test_data)

    for i in range(60, test_len + future_period):
        x_test.append(test_data[i-60:i, 0])
        x_test_temp = np.array(x_test)
        x_test_temp  = np.reshape(x_test_temp , (x_test_temp .shape[0], x_test_temp .shape[1], 1))
        predictions = model.predict(x_test_temp)
        predictions  = np.reshape(predictions , (predictions .shape[0], predictions .shape[1]))
        test_data = np.append(test_data, [predictions[i-60]], axis=0)
        

    predictions = scaler.inverse_transform(predictions)

    return predictions
    

def train(stock, start, end, decimal):
    df = web.DataReader(stock, data_source='yahoo',
                    start=start, end=end)
    
    # Create a new dataframe with only Close
    data = df.filter(['Close'])

    # Convert to numpy
    dataset = data.values

    # Get the number of rows (80% of them)
    training_data_len = math.ceil(len(dataset)* decimal)

    # Scale data for the LSTM model
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(dataset)

    # Creating the training dataset
    train_data = scaled_data[0:training_data_len, :]

    # split the data
    x_train = []
    y_train = []

    for i in range(60, len(train_data)):
        x_train.append(train_data[i-60: i, 0])
        y_train.append(train_data[i, 0])


    # Convert the x_train and y_train to numpy arrays
    x_train, y_train = np.array(x_train), np.array(y_train)

    # Reshape the data to make the array 3D for LSTM model
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    #Build the LSTM model
    model = Sequential()

    model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))

    # Compile the Model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the Model
    model.fit(x_train, y_train, batch_size=1, epochs=1)


    return data, dataset, scaled_data, training_data_len, model, scaler


def plot_past_predictions(stock, start, end):

    data, dataset, scaled_data, training_data_len, model, scaler = train(stock,start, end, 0.8)

    predictions = get_past_predictions(dataset, scaled_data, training_data_len, model, scaler)

    trained = data[:training_data_len]
    valid = data[training_data_len:]
    valid['Predictions'] = predictions


    plt.style.use('dark_background')
    plt.figure(figsize=(8,6))
    plt.title('Model')
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Close Price USD ($)', fontsize=18)
    plt.plot(trained['Close'])
    plt.plot(valid[['Close', 'Predictions']])
    plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
    
    plt.savefig('stocks/Static/charts/'+stock + "_past_predictions")

def plot_future_predictions(stock, start):

    future_period = 365
    start = start
    end = datetime.now()
    data, dataset, scaled_data, training_data_len, model, scaler = train(stock, start, end, 1.0)
    predictions = get_future_predictions(dataset, scaled_data, training_data_len, model, scaler, future_period)


    for i in range(0,future_period):
        last_date = data.index[len(data)-1] + timedelta(days=1)
        data.loc[last_date]= {'Close': None}


    # Plot the data
    trained = data[:training_data_len]
    valid = data[training_data_len:]
    valid['Predictions'] = predictions

    # visualize
    plt.style.use('dark_background')
    plt.figure(figsize=(8,6))
    plt.title('Model')
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Close Price USD ($)', fontsize=18)
    plt.plot(trained['Close'], color="blue")
    plt.plot(valid[['Predictions']], color="red")
    plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')

    plt.savefig('stocks/Static/charts/'+stock + "_future_predictions")












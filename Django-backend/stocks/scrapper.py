import re
import json
import csv
import requests
import os
import math
from io import StringIO
from bs4 import BeautifulSoup
import pandas_datareader as web
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from mpl_finance import candlestick_ohlc
import matplotlib.dates as mpl_dates
from datetime import datetime


url_profile = 'https://ca.finance.yahoo.com/quote/{}/profile?p={}'
url_financials = 'https://ca.finance.yahoo.com/quote/{}/financials?p={}'
url_history = 'https://query1.finance.yahoo.com/v7/finance/download/{}?'


def get_company_profile(stock):
    response = requests.get(url_profile.format(stock, stock))
    soup = BeautifulSoup(response.text, 'html.parser')
    pattern = re.compile(r'\s--\sData\s--\s')
    script_data = soup.find('script', text=pattern).contents[0]

    script_data[:500]
    script_data[-500:]
    start = script_data.find("context")-2
    json_data = json.loads(script_data[start:-12])

    json_data['context'].keys()
    json_data['context']['dispatcher']['stores']['QuoteSummaryStore'].keys()

    # Asset Profile
    assetProfile = json_data['context']['dispatcher']['stores']['QuoteSummaryStore']['assetProfile']

    return assetProfile


def get_price_data(stock):
    response = requests.get(url_profile.format(stock, stock))
    soup = BeautifulSoup(response.text, 'html.parser')
    pattern = re.compile(r'\s--\sData\s--\s')
    script_data = soup.find('script', text=pattern).contents[0]

    script_data[:500]
    script_data[-500:]
    start = script_data.find("context")-2
    json_data = json.loads(script_data[start:-12])

    json_data['context'].keys()
    json_data['context']['dispatcher']['stores']['QuoteSummaryStore'].keys()

    # Price Summary
    price = json_data['context']['dispatcher']['stores']['QuoteSummaryStore']['price']

    return price


def make_company_candle_char(stock, start, end):
    plt.style.use('dark_background')

    plt.figure(figsize=(8, 6))
    data = web.DataReader(stock, data_source='yahoo',
                          start=start, end=end)
    data = data[['Open', 'High', 'Low', 'Close']]

    data.reset_index(inplace=True)

    data['Date'] = data['Date'].map(mdates.date2num)

    ax = plt.subplot()
    # ax.grid(True)
    ax.set_axisbelow(True)
    ax.set_facecolor('black')
    ax.set_title(stock + ' Share Price')
    ax.figure.set_facecolor("#121212")
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    ax.xaxis_date()

    candlestick_ohlc(ax, data.values, width=1, colorup="green")

    # data['SMA5'] = data['Close'].rolling(5).mean()
    # ax.plot(data['Date'], data['SMA5'], color='purple', label='SMA5')

    # plt.legend()

    plt.savefig('stocks/Static/charts/'+stock+ "_candleChart")


def make_company_line_char(stock, start, end):
    plt.style.use('dark_background')


    df = web.DataReader(stock, data_source='yahoo',
                        start=start, end=end)

    plt.figure(figsize=(8, 6))
    plt.title(stock + " Share Price")
    plt.plot(df['Close'], color="green")
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Closing Price', fontsize=18)

    plt.savefig('stocks/Static/charts/'+stock + "_lineChart")


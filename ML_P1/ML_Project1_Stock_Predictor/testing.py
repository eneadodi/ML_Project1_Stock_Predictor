'''This file will be used only to test certain features.'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn
from click._compat import iteritems
import requests
import time
from Web_Stock_Scrapper import StockScraper, StockScraperHelper
import ML_P1_data_extraction as e
import pickle
import os
import threading 
import collections
from pandas.tests.frame.test_sort_values_level_as_str import ascending
from dateutil import rrule
from datetime import datetime, timedelta
import yfinance as yf
from yfinance.utils import auto_adjust


    
def main():
    tickList = 'FOX AHCO SPCE TIGO NLOK UBER DOW FOXA' 
    allTickers = tickList.split()
    yf.download(tickList, start= '2018-08-06',end = '2020-07-07', interval = '1wk',auto_adjust=True)
    queried_data = {}
    for i in allTickers:
        ticker = yf.Ticker(i)
        time.sleep(1)
        print('Curr ticker=  ' + i)
        data = ticker.history(start='2018-08-06', interval = '1wk', end = '2020-08-07', auto_adjust = True)
        print(data.head(10))
        queried_data = {}
        queried_data[i] = {}
        rows = len(data.index)
        for r in range(rows):
            date = str(data.index[r].date())
            queried_data[i][date + ' Close'] = data['Close'].iloc[r]
            queried_data[i][date + ' Volume'] = data['Volume'].iloc[r]
        print("Finished storing Ticker prices for " + i)
        ih = ticker.institutional_holders
        if (ih is None) or (isinstance(ih, list)) or ('Holder' not in ih.columns) :
            queried_data[i]['Institutional Holders'] = []
        else:
            queried_data[i]['Institutional Holders'] = ih['Holder'].tolist()
        
    e.save_obj(queried_data,'MissingTickers')
    
    
    
if __name__ == '__main__':
    main()
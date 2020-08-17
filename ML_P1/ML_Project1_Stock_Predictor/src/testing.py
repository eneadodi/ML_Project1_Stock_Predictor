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
#from sympy import S, symbols, printing

    
def main():
    #df = pd.read_pickle('../data/PRE_OHE_LOW_CRITERIA.pkl')
    df = pd.read_pickle('../data/2YStockDFLowCriteria.pkl')
    #####Useful Filters
    filter_vpw = [col for col in df if ' Volume' in col]
    filter_ppw = [col for col in df if ' Close' in col]
    time_rv = filter_vpw + filter_ppw
    filter_ntrv = [col for col in df if col not in time_rv]
    #
    volume_per_week_df = df[filter_vpw]
    price_per_week_df = df[filter_ppw]
    categorical_values_df = df[filter_ntrv]
    ############################
    
    subs = price_per_week_df.iloc[200]
    
    y = subs.to_numpy()
    x = range(106)
    poly = np.polyfit(x,y,30)
    poly_y = np.poly1d(poly)(x)
    plt.plot(x,poly_y)
    plt.plot(x,y)
    print(y[0:5]) 
    print(poly_y[0:5]) 
    plt.show()
if __name__ == '__main__':
    main()
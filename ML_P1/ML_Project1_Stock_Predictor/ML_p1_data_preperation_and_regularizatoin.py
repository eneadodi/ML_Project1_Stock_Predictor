'''
In ML_P1_data_extraction, I extracted infromation from finviz through Web_Stock_Scrapper module and I queried information from the yfinance API

In this module, I will load in the pickle FullStockData.pkl from the directory, transform this dictionary into a pandas DataFrame,
fix any missing values, normalize/regularize and prepare the data for the ML algorithms.
'''
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


def main():
    
    #Load information
    ss = StockScraper()
    ss.scraped_info = e.load_obj('FullStockData')
    ss.scraped_tickers = ss.extract_tickers()
    stock_information_dict = ss.scraped_info
    
    #Create Pandas
    df = pd.DataFrame(stock_information_dict, columns=stock_information_dict[1600].keys())
    

    print(stock_information_dict[100].keys() == stock_information_dict[1049].keys())
    k = []
    k.append(len(stock_information_dict[0].keys()))
    k.append(len(stock_information_dict[200].keys()))
    k.append(len(stock_information_dict[400].keys()))
    k.append(len(stock_information_dict[600].keys()))
    k.append(len(stock_information_dict[800].keys()))
    k.append(len(stock_information_dict[1000].keys()))
    k.append(len(stock_information_dict[1200].keys()))
    k.append(len(stock_information_dict[1400].keys()))
    k.append(len(stock_information_dict[1600].keys()))
    k.append(len(stock_information_dict[1800].keys()))

    for i in k:
        print(i)
    

if __name__ == "__main__":
    main()
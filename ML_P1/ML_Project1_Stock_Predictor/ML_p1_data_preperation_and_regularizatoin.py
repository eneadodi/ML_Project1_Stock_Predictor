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
import collections
from pandas.tests.frame.test_sort_values_level_as_str import ascending

'''
Yfinance has a lot of missing values. I want to simply print out stats on what values are missing
'''
def get_null_information(filename,df):
    pd.set_option('display.max_rows', len(df))
    f = open(filename + '.txt','w')
    #Divide up the df into three sections: volume per week, price per week, Non time related values
    filter_vpw = [col for col in df if 'Volume' in col]
    filter_ppw = [col for col in df if 'Close' in col]
    time_rv = filter_vpw + filter_ppw
    filter_ntrv = [col for col in df if col not in time_rv]

    volume_per_week_df = df[filter_vpw]
    price_per_week_df = df[filter_ppw]
    categorical_values_df = df[filter_ntrv]
    
    serv = volume_per_week_df.isnull().sum(axis=1)
    serv_m = serv.mean()
    serv_std = serv.std()
    f.write('average # null per ticker for volume columns in dataframe is: ' + str(serv.mean()) + '\n')
    f.write('std # null per ticker for volume columns in dataframe is: ' + str(serv_std) + '\n')
    serp = price_per_week_df.isnull().sum(axis=1)
    serp_m = serp.mean()
    serp_std = serv.std()
    f.write('average # null per ticker for price columns in dataframe is: ' + str(serp.mean())+ '\n')
    f.write('std # null per ticker for price columns in dataframe is: ' + str(serp_std) + '\n')
    serc = categorical_values_df.isnull().sum(axis=1)
    serc_m = serc.mean()
    serc_std = serc.std()
    f.write('average # null per ticker for categorical columns in dataframe is: ' + str(serc.mean())+ '\n')
    f.write('std #null per ticker for categorical columns in dataframe is: ' + str(serc_std) + '\n')
    
    #get subset where the amount of nulls is greater than one standard deviation away from the mean, then sort. 
    #This'll give me an ide aof which stocks to drop.
    serv_outliers = serv[(serv >= serv_m + serv_std)]
    serv_outliers.sort_values(ascending=False,inplace=True,na_position='first')
    
    serp_outliers = serp[(serp >= serp_m + serp_std)]
    serp_outliers.sort_values(ascending=False,inplace=True,na_position='first')
    
    serc_outliers = serc[(serc >= serc_m + serc_std)]
    serc_outliers.sort_values(ascending=False,inplace=True,na_position='first')
    
    f.write("VOLUME NULL OUTLIERS: \n")
    f.write(str(serv_outliers.shape[0]) + "/1859 are one standard deviation away from average null values per ticker\n")
    f.write(serv_outliers.to_string())
    f.write("\n\n\n")
    f.write("PRICE NULL OUTLIERS: \n")
    f.write(str(serp_outliers.shape[0]) +"/1859 are one standard deviation away from average null values per ticker\n")
    f.write(serp_outliers.to_string())
    f.write("\n\n\n")
    f.write("CATEGORICAL NULL OUTLIERS: \n")
    f.write(str(serc_outliers.shape[0]) +"/1859 are one standard deviation away from average null values per ticker\n")
    f.write(serc_outliers.to_string())
    f.write("\n\n\n")
    
    #finally check to see which tickers are featured in all three categories. These stocks probably gotta go.
    serv_i = serv_outliers.index
    serp_i = serp_outliers.index
    serc_i = serc_outliers.index
    f.write("BAD BAD STOCKS")
    bad_stocks = set(serv_i).intersection(serp_i).intersection(serc_i)
    for b in bad_stocks:
        f.write(b)
        f.write('\n-----\n')
    
    pd.reset_option('display.max_rows')
    f.close()
'''
To print all information onto console rather than truncated version. 
'''
def print_full(x):
    pd.set_option('display.max_rows', len(x))
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 2000)
    pd.set_option('display.float_format', '{:20,.2f}'.format)
    pd.set_option('display.max_colwidth', None)
    print(x)
    pd.reset_option('display.max_rows')
    pd.reset_option('display.max_columns')
    pd.reset_option('display.width')
    pd.reset_option('display.float_format')
    pd.reset_option('display.max_colwidth')
    
def check_identical_keys(l):
    k_l = list(l[0].keys())
    
    for i in l[1:]:
        k_i = list(i.keys())
        if collections.Counter(k_l) == collections.Counter(k_i):
            pass
        else:
            print("These lists are not identical")

'''
For columns with missing volume entries, it is good enough to simply put the average volume at these entries. 
'''
def fill_empty_volume_columns(df):
    pass
'''
Calculate average volume of given row.
'''
def calculate_average_volume(row):          
    pass

'''
This one is tricky.
The suboptimal solution i came up with is as follows:
if we are missing price at date X but have the price at both date x-1 and x+1, then fill date x price with:
       x = ((x-1)+(x+1))/2
       
I am making a very strong assumption with this solution, namely, 
I am assuming that the price change between x-1 and x+1 is constant, as in priceChange(x-1,x) = priceChange(x,x+1)
Stocks do not work that way. Between weeks stock prices can fluctuate up and down at a massive scale.
However, I plan to smoothen the curve of prices through time before passing in the values to my ML algorithms
Thus the given solution is acceptable.
'''
def fill_empty_price_columns(df):
    pass

def main():
    
    
    
    #Load information
    ss = StockScraper()
    ss.scraped_info = e.load_obj('FullStockDataVB')
    ss.scraped_tickers = ss.extract_tickers()
    stock_information_dict = ss.scraped_info
    
    #check to see if each dictionary has same key values:
    #check_identical_keys(stock_information_dict)
  
    
    #Create Pandas, reindex, and drop duplicates to support random order and avoid duplicates.
    df = pd.DataFrame(stock_information_dict, columns=stock_information_dict[0].keys())
    df.set_index('Ticker',inplace=True)
    df = df.reindex(np.random.permutation(df.index))
    #print(df.head(5))
    # df.loc[df.astype(str).drop_duplicates().index] There are no duplicates :)
    
    
    
    #print(non_time_values_df.columns)
    #print(volume_per_week_df.columns)
    #print(price_per_week_df.columns)
    
    #print("Describing Volume per Week sub-DataFrame",volume_per_week_df.info())
    #print("Describing Price per week sub-DataFrame",price_per_week_df.info())
    #print("Describing non-time related columns sub-DataFrame",non_time_values_df.info())
    #pd.set_option('display.max_rows', len(df))
    
    get_null_information('NullTickerInformation', df)
    print("donzo")
    
    
if __name__ == "__main__":
    main()
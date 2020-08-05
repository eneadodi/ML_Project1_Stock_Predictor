import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn
import yfinance as yf
from click._compat import iteritems
import requests
import time
from Web_Stock_Scrapper import StockScraper, StockScraperHelper
import pickle
import os
'''
Given a list of ticker names from Stock Scraper, this method will pull information from yfinance API and make it 
ready for Stock Scraper to push into scraped_info variable.
'''
def query_and_format_yfinance_data(allTickers,periodt = '1mo',intervalt = '1wk'):
    s_tickers = ' '.join(allTickers)
    print(s_tickers)
    data = yf.download(tickers = allTickers, period = periodt, interval = intervalt, group_by='ticker', auto_adjust=True,threads= True)
    rows = data.shape[0] - 1
    
    queried_data = {}
    for t in allTickers:
        queried_data[t] = {}
        curr_stock_history = data[t]
        for r in range(rows):
            curr_date = str(curr_stock_history.index[r].date())
            queried_data[t][str(curr_date) + ' Close'] = curr_stock_history['Close'].iloc[r]
            queried_data[t][str(curr_date) + ' Volume'] = curr_stock_history['Volume'].iloc[r]         
            print('Got close and vol for ' + t)
        
        '''
        the following line would through an IndexError exception which I tried to wrap in a 
        'try: except EXCEPTION:' box but it would still for some reason not work. Thus I
        manually changed a line in the yfinance package to check for this error itself:
        self._institutional_holders = holders[1] BECOMES 
        self._institutional_holders = holders[1] if len(holders) > 1 else []
        
        '''
        ih = yf.Ticker(t).institutional_holders 
        if (ih is None) or (isinstance(ih, list)) or ('Holder' not in ih.columns) :
            queried_data[t]['Institutional_Holders'] = []
        else:
            #print(ih['Holder'])
            queried_data[t]['Institutional_Holders'] = ih['Holder'].tolist()
        print('got ih for ' + t)
            
    
    return queried_data

def save_obj(obj,name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)
    
def main():
    
    ss = StockScraper()
    url = 'https://finviz.com/screener.ashx?v=111&'
    #soup = fs.get_entire_HTML_page(url)
    ss.get_all_stock_table_information(url)
    print('got table information')
    #ss.add_RIS()
    #print('added RIS')
    ticker_info_list = ss.scraped_info
    ss.add_all_keys(['book value', 'ForwardEPS','Institutional Holders','Average Volume'] )
    print('added extra Keys')
    ss.scraped_info = ticker_info_list
    ticker_names = ss.scraped_tickers[0:3]
    ##ss.write_info_to_file('stock_info.txt')
    print("scraped info size = " + str(len(ss.scraped_info)))
    print("scraped tickers size = " + str(len(ss.scraped_tickers)))
    d = query_and_format_yfinance_data(ticker_names)
    print(d)
    save_obj(d,"dumbdict2")
    
    
if __name__ == '__main__':
    main()
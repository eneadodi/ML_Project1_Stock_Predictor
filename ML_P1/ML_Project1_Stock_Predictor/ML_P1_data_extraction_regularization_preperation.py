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
import threading 

lock = Lock()
f = open("execution_time_tracking.txt","w")
'''
Given a list of ticker names from Stock Scraper, this method will pull information from yfinance API and make it 
ready for Stock Scraper to push into scraped_info variable.
'''
def query_and_format_yfinance_data(allTickers,periodt = '1mo',intervalt = '1wk'):
    s_tickers = ' '.join(allTickers)
    print(s_tickers)
    data = yf.download(tickers = allTickers, period = periodt, interval = intervalt, group_by='ticker', auto_adjust=True,threads= True)
    rows = data.shape[0]
    
    queried_data = {}
    for t in allTickers:
        queried_data[t] = {}
        curr_stock_history = data[t]
        for r in range(rows):
            curr_date = "Date: " + str(curr_stock_history.index[r].date())
            queried_data[t][str(curr_date) + ' Close'] = curr_stock_history['Close'].iloc[r]
            queried_data[t][str(curr_date) + ' Volume'] = curr_stock_history['Volume'].iloc[r]         
            
        
        '''
        the following line would through an IndexError exception which I tried to wrap in a 
        'try: except EXCEPTION:' box but it would still for some reason not work. Thus I
        manually changed a line in the yfinance package to check for this error itself:
        self._institutional_holders = holders[1] BECOMES 
        self._institutional_holders = holders[1] if len(holders) > 1 else []
        
        '''
        ih = yf.Ticker(t).institutional_holders 
        time.sleep(0.05)
        if (ih is None) or (isinstance(ih, list)) or ('Holder' not in ih.columns) :
            queried_data[t]['Institutional_Holders'] = []
        else:
            
            queried_data[t]['Institutional_Holders'] = ih['Holder'].tolist()
        
            
    print("finished gathering ticker info for this quarter")
    return queried_data

def save_obj(obj,name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

def prepare_Stock_Scraper():
    ss = StockScraper()
    url = 'https://finviz.com/screener.ashx?v=111&'

    ss.get_all_stock_table_information(url)
    print('got table information')
    ss.add_RIS()
    print('added RIS')
    return ss

def query_and_save_yf(ss,t_n,picklename):
    t_start = time.time()
    d = query_and_format_yfinance_data(t_n)
    t_end = time.time()
    t_total = t_end - t_start
    yfinance_query_time = "query_and_format_yfinance_data for this quarter of ticker names took " + "{:.4f}".format(t_total) + " seconds\n"
    f.write(yfinance_query_time)
    lock.acquire() #to ensure thread synchronization is well met.
    ss.add_all_specified_key_value_pair(d)
    progress_save_p = ss.scraped_info
    lock.release()
    save_obj(progress_save_p, picklename)
    print('Saved to ' + picklename + '\n')
    time.sleep(0.5)
    
def main():
    
    
    '''
    Scraping information from Finviz first
    '''
    t_start = time.time()
    ss = prepare_Stock_Scraper()
    t_end = time.time()
    
    t_total = t_end - t_start
    finviz_scraping_time = "prepare_Stock_Scraper() took " + "{:.4f}".format(t_total) + " seconds\n"
    f.write(finviz_scraping_time)
    
    ticker_info_list = ss.scraped_info
    ticker_names = ss.scraped_tickers
    
    
    divisor = len(ticker_names) // 4
    
    t_n1 = ticker_names[:divisor]
    t_n2 = ticker_names[divisor:(divisor*2)]
    t_n3 = ticker_names[(divisor*2):(divisor*3)]
    t_n4 = ticker_names[(divisor*3):]
    
    thread1 = threading.Thread(target=query_and_save_yf,args=(ss,t_n1,'quarterStockData',))
    
    thread2 = threading.Thread(target=query_and_save_yf,args=(ss,t_n2,'HalfStockData',))
    
    thread3 = threading.Thread(target=query_and_save_yf,args=(ss,t_n3,'ThreeQuarterStockData',))
    
    thread4 = threading.Thread(target=query_and_save_yf,args=(ss,t_n4,'FullStockData',))
    
    thread1.start()
    thread2.start()
    thread3.start()
    thread4.start()
    
    '''
     query_yf(ss,t_n1,'quarterStockData')
    
    query_yf(ss,t_n2,'HalfStockData')
    
    query_yf(ss,t_n3,'ThreeQuarterStockData')
    
    query_yf(ss,t_n4,'FullStockData')
    
    d2 = query_and_format_yfinance_data(t_n2)
    ss.add_all_specified_key_value_pair(d2)
    progress_save_p2 = ss.scraped_info
    save_obj(progress_save_p2, "halfStockData")
    print('Saved this quarter of complete dictionary')
    
    d3 = query_and_format_yfinance_data(t_n3)
    ss.add_all_specified_key_value_pair(d3)
    progress_save_p3 = ss.scraped_info
    save_obj(progress_save_p3, "ThreeQuarterStockData")
    print('Saved this quarter of complete dictionary')
    
    d4 = query_and_format_yfinance_data(t_n4)
    ss.add_all_specified_key_value_pair(d4)
    progress_save_p4 = ss.scraped_info
    save_obj(progress_save_p4, "FullStockData")
    print('Saved this quarter of complete dictionary')
    '''
    
    
if __name__ == '__main__':
    main()
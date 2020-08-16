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
from pandas.tests.scalar.interval.test_interval import interval


#lock = threading.Lock()
f = open("execution_time_tracking_final.txt","a")


'''
Given a list of ticker names from Stock Scraper, this method will pull information from yfinance API and make it 
ready for Stock Scraper to push into scraped_info variable.
'''
''' 
def query_and_save_yf2(ss,picklename):
    f = open('ConcludedStocksYF2.txt','a')
    ticker_names = ss.scraped_tickers
    for i in ticker_names:
        print('Curr Ticker: ' + i)
        ticker = yf.Ticker(i)
        time.sleep(0.2)
        data = ticker.history(period = '5y', interval = '1wk',auto_adjust=True)
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
        
        ss.add_all_specified_key_value_pair(queried_data)
        progress_save_p = ss.scraped_info
        
        e.save_obj(progress_save_p,picklename)
        print("FInished completely with Ticker: " + i)
        f.write("Finished: " + i + '\n')
    f.close()
    
def main():
    print('start!')
    time.sleep(2)
    ss = StockScraper()
    ss.scraped_info = e.load_obj('preYFinanceData')
    ss.scraped_tickers = ss.extract_tickers()

    query_and_save_yf2(ss,"FullStockDataVC")
'''

'''
Given a list of ticker names from Stock Scraper, this method will pull information from yfinance API and make it 
ready for Stock Scraper to push into scraped_info variable.
'''
def query_and_format_yfinance_data(allTickers,periodt = '1mo',intervalt = '1wk'):
    print('period = ', periodt)
    print('interval = ', intervalt)
    s_tickers = ' '.join(allTickers)
    print(s_tickers)
    data = yf.download(tickers = allTickers, period = periodt, interval = intervalt, group_by='ticker', auto_adjust=True,threads= True)
    rows = data.shape[0]
    
    queried_data = {}
    for t in allTickers:
        print("Curr ticker: " + str(t))
        queried_data[t] = {}
        curr_stock_history = data[t]
        for r in range(rows):
            curr_date = "Date: " + str(curr_stock_history.index[r].date())
            queried_data[t][str(curr_date) + ' Close'] = curr_stock_history['Close'].iloc[r]
            queried_data[t][str(curr_date) + ' Volume'] = curr_stock_history['Volume'].iloc[r]         
            
        time.sleep(0.1)
        '''
        the following line would through an IndexError exception which I tried to wrap in a 
        'try: except EXCEPTION:' box but it would still for some reason not work. Thus I
        manually changed a line in the yfinance package to check for this error itself:
        self._institutional_holders = holders[1] BECOMES 
        self._institutional_holders = holders[1] if len(holders) > 1 else []
        
        '''
        '''ih = yf.Ticker(t).institutional_holders 
        time.sleep(0.2)
        if (ih is None) or (isinstance(ih, list)) or ('Holder' not in ih.columns) :
            queried_data[t]['Institutional_Holders'] = []
        else:
            
            queried_data[t]['Institutional_Holders'] = ih['Holder'].tolist()
        '''
            
    print("finished gathering ticker info from yfinance")
    return queried_data

def save_obj(obj,name):
    with open(name + '.pkl', 'wb') as fi:
        pickle.dump(obj, fi, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open(name + '.pkl', 'rb') as fi:
        return pickle.load(fi)

def prepare_Stock_Scraper():
    ss = StockScraper()
    url = 'https://finviz.com/screener.ashx?v=111&'

    ss.get_all_stock_table_information(url,market_cap='100000000',minPrice =2,maxPrice = 2000,volume=100000)
    print('got table information')
    ss.add_RIS()
    print('added RIS')
    return ss

def query_and_save_yf(ss,t_n,picklename):
    t_start = time.time()
    d = query_and_format_yfinance_data(t_n,periodt='5y',intervalt='1wk')
    t_end = time.time()
    t_total = t_end - t_start
    yfinance_query_time = "query_and_format_yfinance_data for this quarter of ticker names took " + "{:.4f}".format(t_total) + " seconds\n"
    print('starting adding yfinance info')
    ss.add_all_specified_key_value_pair(d)
    progress_save_p = ss.scraped_info
    print('ending adding yfinance info')
    #lock.release()
    save_obj(progress_save_p, picklename)
    print('Saved to ' + picklename + '\n')
    print(yfinance_query_time)
    f.write(yfinance_query_time)
    #lock.acquire() #to ensure thread synchronization is well met.
    #time.sleep(0.5)

def main():
    
    
    '''
    Scraping information from Finviz first
    '''
    
    '''t_start = time.time()
    ss = prepare_Stock_Scraper()
    t_end = time.time()
    
    t_total = t_end - t_start
    finviz_scraping_time = "prepare_Stock_Scraper() took " + "{:.4f}".format(t_total) + " seconds for the larger pool\n"
    f.write(finviz_scraping_time)
    save_obj(ss.scraped_info,'maxScrapExperiment')
    '''
    ss = StockScraper()
    ss.scraped_info = load_obj('maxScrapExperiment')
    ss.scraped_tickers = ss.extract_tickers()
    print('size of tickers = ', len(ss.scraped_tickers))
    ticker_names = ss.scraped_tickers
    ticker_info = ss.scraped_info
    
    save_obj(ticker_info, "preYFinanceDataLowCriteria")
    
    query_and_save_yf(ss, ticker_names, "FullStockDataLowCriteria")
    '''
    query_and_save_yf(ss,t_n1,'1.5StockData')
    
    query_and_save_yf(ss,t_n2,'2.5StockData')

    query_and_save_yf(ss,t_n3,'3.5QuarterStockData')
    
    query_and_save_yf(ss,t_n4,'4.5StockData')
    
    query_and_save_yf(ss,t_n5,'FullStockData')
    '''
    f.close()

    '''
    VERSION B: Threading made yfinance go bonkers therefore i opted out.
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
    
    thread1.join()
    thread2.join()
    thread3.join()
    thread4.join() 
    '''
    
if __name__ == '__main__':
    main()
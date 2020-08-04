import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn
import yfinance as yf
from click._compat import iteritems
import requests
import time
from Web_Stock_Scrapper import StockScraper, StockScraperHelper

'''
Given a list of ticker names from Stock Scraper, this method will pull information from yfinance API and make it 
ready for Stock Scraper to push into scraped_info variable.
'''
def query_and_format_yfinance_data(allTickers,periodt = '1mo',intervalt = '1wk'):
    s_tickers = ' '.join(allTickers[0:10])
    print(s_tickers)
    data = yf.download(tickers = allTickers, period = periodt, interval = intervalt, group_by='ticker', auto_adjust=True,threads= True)
    return data

def main():
    ss = StockScraper()
    url = 'https://finviz.com/screener.ashx?v=111&'
    #soup = fs.get_entire_HTML_page(url)
    ss.get_all_stock_table_information(url)
    print('got table information')
    ss.add_RIS()
    print('added RIS')
    ticker_info_list = ss.scraped_info
    ss.add_all_keys(['book value', 'ForwardEPS','Institutional Holders','Average Volume'] )
    print('added extra Keys')
    ss.scraped_info = ticker_info_list
    ticker_names = ss.scraped_tickers
    ##ss.write_info_to_file('stock_info.txt')
    print("scraped info size = " + str(len(ss.scraped_info)))
    print("scraped tickers size = " + str(len(ss.scraped_tickers)))
    df = query_and_format_yfinance_data(ticker_names,periodt='2yr')
    print('queried data')
    print(type(df))
    print(df.shape)
    df.head(5).apply(print)
    #book value, forwardEPS, averageVolume (3 month), threeYearAverageReturn ,Two Hundred Day Average

if __name__ == '__main__':
    main()
'''
Created on Jul 26, 2020

@author: Enea Dodi
'''

from bs4 import BeautifulSoup, SoupStrainer, Tag
import requests
import lxml
import re 
import urllib
import textwrap
from nltk import tokenize
import constants
import numpy as np
import os


'''
To not overburden the StoclScraper class with methods, I dedicated a helper class which does all formatting jobs for 
the StockScraper class.
This class will do things such as:
    removing data
    adding data
    formatting data to pass to StockScraper methods
'''
class StockScraperHelper(object):
    '''
    Constructor
    '''
    def __init__(self):
        pass
    
    '''
    A quick regex method where it translates large number abbreviations to true number:
    ex: 100k -> 100,000
        1m -> 1,000,000
        2b -> 2,000,000,000
        - -> nan (if value is missing)
    a modification from https://gist.github.com/gajeshbhat/67a3db79a6aecd1db42343190f9a2f17
    '''
    def nabbr_to_number(self,x):
        if x == '-':
            return np.nan
        rnum = 0
        num_map = {'K':1000, 'M':1000000, 'B':1000000000}
        if x.isdigit():
            rnum = int(x)
        elif ',' in x:
            rnum = float(x.replace(',',''))
        else:
            if len(x) > 1:
                rnum = float(x[:-1]) * num_map.get(x[-1].upper(), 1)
        return int(rnum+0.5)

    def remove_empty(self,row):
        for x in row:
            if x == '-':
                x = np.nan

    def float_convert(self,val):
        if val == '-':
            return np.nan
        else:
            return float(val)
    def row_to_dict(self,row):
        del row[0]
        del row[1]
        del row[-4]
        self.remove_empty(row)
        return {'Ticker':row[0],'Sector':row[1],'Industry':row[2],'Country':row[3],'Market Cap': self.nabbr_to_number(row[4]),
                'Price':self.float_convert(row[5]),'Change':self.float_convert(row[6].strip('%'))/100,'Volume': self.float_convert(row[7].replace(',',''))}

    
    def print_tickers(self,scraped_ticks):
    
        for i in scraped_tickers:
            print(i)
'''
This class will be used to scrape information from Finviz.com and extract information from yfinance.
'''
class StockScraper(object):
    
    '''
    Constructor
    '''
    def __init__(self):
        self.helper = StockScraperHelper()
        self.scraped_info = []
        self.scraped_tickers = []
        self.HEADERS = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/71.0.3578.98 Safari/537.36'}
        #Used to prevent authorization issues.
    
    """
    param: url
    return the Entire HTML code fo the website
    """
    def get_entire_HTML_page(self,url):
        page = requests.get(url,headers=self.HEADERS).text
        soup = BeautifulSoup(page,features ="lxml")
        return soup
    
        
    '''
    provided the soup of finviz.com/screener.ashx?v=111 this method will output a list of dictionaries provided in format:
    Stock = {'Ticker': , 'Company': , 'Sector': , 'Industry':, 'Country':, 'Market Cap': , 'Price' , 'Change', 'Volume' } 
    
    Parameters:   
        soup - the lxml soup of the page
        sector - Filter for certain sectors. Default 'all'
        industry - Filter for certain industry. Default 'all'
        country - Filter for certain country. Default 'all'
        market_cap - Fitler for certain Market Cap. Default 'all'
        minPrice - Filter for certain minimum Price. Default 8
        maxPrice - Filter for certain maximum Price. Default 1000
        volume - Filter for certain Volume. Default 200,000
    '''
    def get_stock_table_information(self,soup,sector='all',industry='all',country='all',market_cap='all',minPrice=8,maxPrice=1000,volume=200000):
        table_rows = soup.find_all('tr',{'class':'table-dark-row-cp'}) + soup.find_all('tr',{'class':'table-light-row-cp'}) 
        stock_list = []
        for r in table_rows:
            info = []
            for child in r.descendants:
                if child.name == 'a':
                    info.append(child.text)
            r_dict = self.helper.row_to_dict(info)
            stock_list.append(r_dict)

        if sector != 'all':
            stock_list = list(filter(lambda x: x['Sector'] == sector,stock_list))
        if industry != 'all':
            stock_list = list(filter(lambda x: x['Industry'] == industry,stock_list))
        if country != 'all':
            stock_list = list(filter(lambda x: x['Country'] == country,stock_list))
        if market_cap != 'all':
            stock_list = list(filter(lambda x: x['Market Cap']  > int(market_cap),stock_list))
        stock_list = list(filter(lambda x: (x['Price'] > minPrice) & (x['Price'] < maxPrice),stock_list))
        return stock_list
    
    
    def write_IH_to_file(self,filename):
        abs_path =  'C:/Users/Enea Dodi/git/ML_P1/ML_Project1_Stock_Predictor/' + filename
        f = open(abs_path,'w')
        for s in self.scraped_info: #write every stock dictionary in list to file
            f.write(str(s['Institutional_Holders']) + '\n')
        f.close()
        return
    
    def write_info_to_file(self,filename):
        abs_path =  'C:/Users/Enea Dodi/git/ML_P1/ML_Project1_Stock_Predictor/' + filename
        f = open(abs_path,'w')
        for s in self.scraped_info: #write every stock dictionary in list to file
            f.write(str(s) + '\n\n\n\n')
        f.close()
        return
    
    '''
    provided the soup of finviz.com/screener.ashx?v=111 this method will iteratively call get_stock_table_information for each page available.
    Because of the constant flux of amount of Tickers available on a free Finviz account, it'll first find count of total
    tickers, then integer divide it to calculate how many iterations of get_stock_table_information() this method will call.
    
    This will return a large dictionary of dictionaries in format:
    Stock = {'Ticker': , 'Company': , 'Sector': , 'Industry':, 'Country':, 'Market Cap': , 'Price' , 'Change', 'Volume' } 
    
    Parameters:
        soup - the lxml soup of the page
        sector - Filter for certain sectors. Default 'all'
        industry - Filter for certain industry. Default 'all'
        country - Filter for certain country. Default 'all'
        market_cap - Fitler for certain Market Cap. Default 'all'
        minPrice - Filter for certain minimum Price. Default 8
        maxPrice - Filter for certain maximum Price. Default 1000
        volume - Filter for certain Volume. Default 200,000
    '''
    def get_all_stock_table_information(self,url,sector='all',industry='all',country='all',market_cap='all',minPrice=2.50,maxPrice=2000,volume=200000,minimal = True):
        '''
        First we get soup and find the td with class 'count-text'. This'll give us the total number of
        tickers.
        As each page has 20 listings, we do integer divison
        '''
        soup = self.get_entire_HTML_page(url)
        total = int(soup.find('td',{'class':'count-text'},recursive=True).text.split(' ')[1])
        iterations = total // 20
        
        url_extension = 'r='
        curr_tickers = 21
        
        l = self.get_stock_table_information(soup, sector, industry, country, market_cap, minPrice, maxPrice, volume)
        
        for i in range(iterations):
            next_url = url + url_extension + str(curr_tickers)
            print(next_url)
            next_soup = self.get_entire_HTML_page(next_url)
            curr_tickers += 20
            l = l + self.get_stock_table_information(next_soup, sector, industry, country, market_cap, minPrice, maxPrice, volume)
        
        if minimal:
            '''
            There were many ways for me to implement the minimal parameter. While seemingly a longer process, 
            due to the filterations done by the other parameters, it is best to remove the specified columns at the end.
            '''
            keys = ('Market Cap', 'Price', 'Change', 'Volume')
            for i in l:
                for k in keys:
                    del i[k]
        
        
        #filename = 'stock_info.txt'
        #self.write_list_to_file(filename,l)
        #print(l)
        self.scraped_info = l
        self.scraped_tickers = self.extract_tickers()

    '''
    Finviz offers in depth analysis for each stock on their individual page. I will be scraping three additional features for the ML algorithm:
    Reccomendation : the average analyst recommendation (1 best, 5 worst)
    Income : income of company
    Sales : Sales of company
    '''
    def add_RIS(self):
        url = 'https://finviz.com/quote.ashx?t='
        for i in range(len(self.scraped_tickers)):
            t_r = self.get_entire_HTML_page(url+self.scraped_tickers[i]).find('table',{'class':'snapshot-table2'},recursive = True).find_all('tr',{'class':'table-dark-row'})

            income_r = t_r[2].find_all('td')[1].text
            sales_r = t_r[3].find_all('td')[1].text
            recc_r = t_r[-1].find_all('td')[1].text
            
            income = self.helper.nabbr_to_number(income_r) if income_r != '-' else np.nan
            sales = self.helper.nabbr_to_number(sales_r) if sales_r != '-' else np.nan
            recc = self.helper.nabbr_to_number(recc_r) if recc_r != '-' else np.nan
            
            self.scraped_info[i]['Income'] = income 
            self.scraped_info[i]['Sales'] = sales 
            self.scraped_info[i]['Recommendations'] = recc
            
    
    def extract_tickers(self):
        return list(map(lambda x: x['Ticker'], self.scraped_info))
    
    def add_all_keys(self,key_vals):
        for t in self.scraped_info:
            for new_k in key_vals:
                t[new_k] = None
    
    def add_all_same_key_value_pairs(self,key_vals):
        for t in self.scraped_info:
            for new_k,new_v in key_vals.items():
                t[new_k] = new_v
    
    '''
    Add a key, value pair to certain index of info_scraped
    '''
                
    def add_specified_key_value_pair(self,tname,key,val):
        t = next(item for item in self.scraped_info if item["Ticker"] == tname)
        t[key] = val
    
    '''
    In this case, we want to automate the add_specified_key_value_pair for all
    stocks in the info_scraped list. To do this we pass in a dictionary with the 
    following format:
    
    d = {
            'TICKERNAME1' : {'NEWKEY1A' : 'NEWVAL1A',
                             'NEWKEY1B' : 'NEWVAL1B',
                             ....
                            }
            'TICKERNAME2' : {'NEWKEY1A' : 'NEWVAL1A',
                             'NEWKEY1B' : 'NEWVAL1B',
                             ....
                            }
            ....
        }
        
    a function which will format this information in the specified order will likely be necassary
    '''
    def add_all_specified_key_value_pair(self,d):
        for tickn,kickd in d.items():
            for k,v in kickd.items():
                self.add_specified_key_value_pair(tickn, k, v)
    
    
    def print_tickers(self):
        self.helper.print_tickers(self.scraped_tickers)
        for i in scraped_tickers:
            print(i)

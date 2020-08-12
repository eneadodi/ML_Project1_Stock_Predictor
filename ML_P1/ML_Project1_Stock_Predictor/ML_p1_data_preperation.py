'''
In ML_P1_data_extraction, I extracted infromation from finviz through Web_Stock_Scrapper module and I queried information from the yfinance API

In this module, I will load in the pickle FullStockData.pkl from the directory, transform this dictionary into a pandas DataFrame,
fix any missing values, normalize/regularize and prepare the data for the ML algorithms.
'''
import math
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
import copy
from pickle import HIGHEST_PROTOCOL
from pandas.tests.reshape.test_pivot import dropna
from numpy import NaN
'''
Yfinance gave many null dates that were between the weekly intervals. As in, 
if a week time is 2020-07-06 to 2020-07-13, then any values that were between these two dates are all null for all stocks. 
Thus we need to clean up that information.
First week is 2018-08-06 and last week is 2020-08-06
'''
def pick_date_range(df,startd,endd):
    start = datetime(startd[0],startd[1],startd[2])
    end = datetime(endd[0],endd[1],endd[2])
    
    l = []
    l.extend(['Ticker','Sector','Industry','Country','Income','Sales','Recommendations','Institutional Holders','Institutional Holders'])
    for dt in rrule.rrule(rrule.WEEKLY,dtstart=start,until=end):
        #l.append("Date: " + str(dt.date()) + " Close")
        l.append(str(dt.date()))
    cols = [c for c in df.columns if any(map(c.__contains__, l))]
    return df[cols]
    
'''
Yfinance has a lot of missing values. I want to simply print out stats on what tickers are outside 2*std null values from mean amount of null values.
This method also removes the worst tickers if specified. Default it does not.
If removing tickers, then all tickers with more than 2*std null values from mean amount of null values get deleted from the dataframe.
'''
def get_null_information(filename,df,remove_worst = False):
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
    serv_outliers = serv[(serv >= serv_m + 2*serv_std)]
    serv_outliers.sort_values(ascending=False,inplace=True,na_position='first')
    
    serp_outliers = serp[(serp >= serp_m + 2*serp_std)]
    serp_outliers.sort_values(ascending=False,inplace=True,na_position='first')
    
    serc_outliers = serc[(serc >= serc_m + 2*serc_std)]
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
    
    if remove_worst == True:
        for i in serp_i:
            df.drop(i,inplace=True)
        return df
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
    filter_vpw = [col for col in row if 'Volume' in col]  
    row_v = row[filter_vpw]
    m = row_v.mean(axis=1)
    return m[0]

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

'''
Upon Inspection of the Excel and null values of the Dataframe,
There was no need of the previous three methods: fill_empty_volume_columns, calculate_average_volume, fill_empty_price_columns
This is because if there were empty columns, then that was due to the Stock not existing at Date of column or 
(for very few) yfinance didn't give data. 
IT IS FROWNED UPON to give magic values for features in a ML algorithm, and if the empty cells were surrounded by 
full cells, then updating the empty cells with one of the aforementioned methods would make sense. But in these cases,
we would be giving the stocks magical values before they actually existed

Thus because of the low density of empty cells, I will simply fill them by hand :)
TERP and LM in 2020-08-03 were merged thus i will simply put the previous price and average volume.
'''
def fill_empty_date_cells(df):
    df.at['MAIN','Date: 2018-08-20 Volume'] = calculate_average_volume(df.loc[['MAIN']])
    df.at['MAIN','Date: 2018-08-20 Close'] = 34.925841473925
    df.at['LM','Date: 2020-08-03 Volume'] = calculate_average_volume(df.loc[['LM']])
    df.at['LM','Date: 2020-08-03 Close'] = 49.98
    df.at['TERP','Date: 2020-08-03 Volume'] = calculate_average_volume(df.loc[['TERP']])
    df.at['TERP','Date: 2020-08-03 Close'] = 18.85
    print('Done filling empty cells.')
    
def initalStart():     
    #Load information
    ss = StockScraper()
    ss.scraped_info = e.load_obj('data/FullStockDataVB')
    stock_information_dict = ss.scraped_info
    ss.scraped_tickers = ss.extract_tickers()
    
    #check to see if each dictionary has same key values:
    #check_identical_keys(stock_information_dict)
  
    
    #Create Pandas, reindex, and drop duplicates to support random order and avoid duplicates.
    df = pd.DataFrame(stock_information_dict, columns=stock_information_dict[0].keys())
    df.set_index('Ticker',inplace=True)
    df = df.reindex(np.random.permutation(df.index))
    #df.loc[df.astype(str).drop_duplicates().index] There are no duplicates :)
    
    
    
    #I have yet to decide if my Machine Learning Algorithm will be learning from a 2 year time span or a 5 year time span.
    # Thus I will make a copy of both options
    dstart2y = [2018,8,6]
    dstart5y = [2015,8,3]
    dend = [2020,8,7]
    for i in df.columns:
        print(i)

    df2y = pick_date_range(df,dstart2y,dend)
    #df5y = pick_date_range(df, dstart5y, dend)
    df2y.to_pickle('2YStockDFB',protocol=HIGHEST_PROTOCOL)
    #df5y.to_pickle('5YStockDFB',protocol=HIGHEST_PROTOCOL)
    
def remove_incomplete_rows(df):
    counter = 0
    for index, row in df.iterrows():
        if pd.isnull(row['Date: 2018-08-13 Close']):
            print(index)
            counter += 1
            df.drop(index=index,inplace=True)
            
    print('COunter is ' , counter)

'''
This function will be used after the fill_null_income function so
we suppose all income values are defined. This order was to optimize
income by using real sales listed whenever we can, and to switch
the optimization to most average for the candidates with both
Sales and Income missing
NOTE: This comes with the presupposition that income is a better 
feature than sales.

For Sales, we find the mean/median ratio difference between known sales and incomes for Sectors,
Then apply the listed income to a function including this ratio to spit out a solid
variantable and averaged sales value.
'''
def fill_null_Sales(df):
    in_df_ind = df.groupby(df['Industry'])
    in_df_sect = df.groupby(df['Sector'])
    df_null = df[df['Sales'].isnull()]
    count = 0
    for index, r in df_null.iterrows():
        count+=1
        sector_n = r['Sector']
        industry_n = r['Industry']
        income_v = r['Income']
        s = in_df_ind.get_group(industry_n).mean()
        sector_or_ind_sales = s.Sales
        if math.isnan(sector_or_ind_sales):
            sector_or_ind_sales = in_df_sect.get_group(sector_n).mean().Sales
        sector_income = s.Income
        ratio_si = sector_or_ind_sales / sector_income
        print('Income value for ticker: ', r.name, ' is: ', income_v)
        print('Average sales for this sector is: ', sector_or_ind_sales,' For sector ', industry_n)
        print('Average income for this sector is: ', sector_income,' For sector ', industry_n)
        print('Thus ratio is ' , ratio_si)
        print('new_sales_val is  ', abs(ratio_si *  income_v))
        new_sale_val = abs(ratio_si *  income_v)
        print('New Sales value for ticker: ', r.name, ' is: ', new_sale_val,'\n\n')
        df.loc[index,'Sales'] = new_sale_val
    print('count is : ', count)

    
    
'''
Arguably if income is null, this is an important feature for predicting stock prices thus some may want to simply drop the items that do not feature
income. However, for the sake of learning, I will develop the method which will fill these values

The income missing value method will follow this heuristic:

If Sales Column is not NULL:
    Group by sector, and divide income by sale. Then take this value and multiply the sales column to get the income column. 
    make sure to not multiply negative by negative.

If Sales Column is NULL:
    Group by industry, get mean value. 

Error checked by calculating the mean value given to fill null incomes by the if and the else. If the mean value was roughly the same, then math was done right.
'''
def fill_null_Income(df):
    in_df_indust = df.Income.groupby(df['Industry'])
    count = 0
    in_df_sec = df.groupby(df['Sector'])
    
    df_null = df[df['Income'].isnull()]
    for index, r in df_null.iterrows():
        count += 1
        print('Currently working on ', r.name)
        sales = r['Sales']
        print('sales is: ',sales)
        if math.isnan(sales):
            industry = r['Industry']
            m_val_by_industry = in_df_indust.get_group(industry).mean()
            
            if math.isnan(m_val_by_industry):
                sector = r['Sector']
                m_val_by_sector = in_df_sec.get_group(sector).Income.mean()
                print('Entered first if, value given is: ' , m_val_by_sector)
                df.loc[index,'Income'] = m_val_by_sector
            else:
                df.loc[index,'Income'] = m_val_by_industry
                print('Entered first if, value given is: ' , m_val_by_industry)
        else:#Sales isn't none
            sector = r['Sector']
            df_sect = in_df_sec.get_group(sector)
            median_income_div_sales = (df_sect['Income'] / df_sect['Sales']).median()# because very sucessful companies can scew
            #print('med ot is ', median_income_div_sales) 
            #print(df_sect.head())
            fill = 0
            if median_income_div_sales < 0:
                fill = -1*abs(median_income_div_sales * sales)
                df.loc[index,'Income'] = fill
            else:
                fill = abs(median_income_div_sales * sales)
                df.loc[index,'Income'] = fill
            print('Entered second if, value given is: ', fill)
    print(count, ' is count')
    
    
'''
Popped as an idea in my head. Seemed useful.
'''
global_count = 0 
def print_global_count():
    global global_count
    print(global_count)
    global_count+=1

def reset_global_counter():
    global global_count
    global_count = 0


'''
A general class to one_hot_encode. 
df = pandas DataFrame to be morphed
c_name = column name to one hot encode
contains_null = if dataframe column contains null, then add a new column
                for tracking which ones contained a null in the first place
null_value = so user can decide on what the 0 value for the one hot encoding should be:
            ex: null_value = [] , null_value = Circle(radius=1), etc
'''
def one_hot_encode(df,c_name,contains_null = False,null_value = None,sparsev=False):
    r_df = pd.concat([df,pd.get_dummies(df[c_name],prefix=c_name,sparse=sparsev)],axis=1,sort=False)
    
    '''
    Quick way to make a boolean contains column
    '''
    def check_for_null(x):
        
        if null_value is None:
            if pd.isnull(x):
                print_global_count() # Recommendations should be 55
                return 0
            else:
                return 1
        else:
            if x is null_value:
                return 0
            else:
                return 1
            
            
    if contains_null == True:
        r_df['Contain ' + c_name] = df[c_name].apply(check_for_null)
    
    r_df.drop([c_name],axis=1,inplace=True)
    print('Done one hot encoding: ' , c_name)

    reset_global_counter()
    return r_df
    
def main():
    
    f = open('XXXXXX.txt','a')
    #####Will only be called once, when we have the dictionary but not pandas.
    #initalStart()
    ############################
    
    
    
    
    df = pd.read_pickle('data/2YStockDFBcleaner.pkl')
    #df5y = pd.read_pickle('5yStockDF')
    #df.to_excel('bipbapboop4.xlsx')
    
    
    #####Useful Filters
    #filter_vpw = [col for col in df if 'Volume' in col]
    #filter_ppw = [col for col in df if 'Close' in col]
    #time_rv = filter_vpw + filter_ppw
    #filter_ntrv = [col for col in df if col not in time_rv]
    #
    #volume_per_week_df = df[filter_vpw]
    #price_per_week_df = df[filter_ppw]
    #categorical_values_df = df[filter_ntrv]
    ############################
    
    
    
    
    ####USED TO REMOVE WORST TICKERS (tickers with A LOT of missing features)
    #get_null_information('NullTickerInformation2YB', df,remove_worst=False)
    #new_df.to_pickle('2YStockDFBclean',protocol=HIGHEST_PROTOCOL)
    #new_df.to_excel('2YStockDFBcleanexcel.xlsx')
    ############################
    
    
    
    
    ####To get information on what values are still mising or NULL
    #ndf = categorical_values_df.isnull().sum(axis=0)
    #print_full(ndf)
    ############################
    
    
    
    
    
    #print_full(ih.head(20))
    ####USED TO FILL NULL INCOME VALUES WITH APPROPRIATE VALUES
    #fill_null_Income(df)
    #categorical_values_df.to_excel('boopdiboop.xlsx')
    #e.save_obj(df,'2YStockDFBcleanerA')
    #Pre method income avg: 769344444.4
    #Post method income avg: 783475326.4
    #############################
    
    
    
    
    
    ###Used to fill NULL SALES VALUES WITH APPROPRIATE VALUES 
    #fill_null_Sales(df)
    #categorical_values_df.to_excel('boopdibab.xlsx')
    #e.save_obj(df, '2YStockDFBcleanerB')
    #Pre method Sales avg: 11688020115
    #Post method Sales avg: 11762226708
    #############################
    
    
    
    
    
    ###REMOVE TICKERS THAT BELONG TO SPECIFIED Industries, Countries
    '''
    This is important because if a feature doesn't appear enough, then the ML may either not know what to do with the information
    Or incorrectly use the information. I fear it incorrectly using information when there are One-Hot-Encoded columns with only one or two True values
    It hurts removing then along with the 20 stocks that go with them but for the sake of Occam's razor I will do it. Only those that appear less than three time
    are removed. 
    A total of 41 stocks are getting removed :(
    '''
    #bad_industries = ['Uranium','Textile Manufacturing', 'Real Estate - Diversified','Publishing','Pollution & Treatment Controls','Pharmaceutical Retailers','Paper & Paper Products',
    #                  'Other Precious Metals & Mining','Oil & Gas Drilling','Marine Shipping','Luxury Goods','Financial Conglomerates','Exchange Traded Fund',
    #                  'Electronics & Computer Distribution','Copper','Conglomerates','Business Equipment & Supplies','Aluminum']
    #bad_countries = ['Taiwan','Spain','Russia','Philippines','Peru','Panama','Norway','Italy','Indonesia','Colombia','Cayman Islands','Sweden']
    # 
    #df_bit = df[df['Industry'].isin(bad_industries)].index
    #print('IND length is ', len(df_bit))
    #df.drop(labels=df_bit,axis=0,inplace=True)
    #
    #df_bct = df[df['Country'].isin(bad_countries)].index
    #print('COUN length is ', len(df_bct))
    #df.drop(labels=df_bct,axis=0,inplace=True)
    #############################
    
    
    
    
    
    ###USED TO MAKE ONE HOT ENCODING OF RECOMMENDATOINS
    #df = one_hot_encode(df,c_name = 'Recommendations',contains_null=True)
    #oh_names = df.iloc[:,-6:].columns #one hot names
    #index_oh = 5
    #for n in oh_names: # To move to position i'd like
    #   df.insert(index_oh,n,df.pop(n))
    #    index_oh+= 1
    #df.drop(columns='Recommendations_5.0',inplace=True) # If not 1-4, then must be 5, Also there is only one such
    #df.to_excel('bipbapboop3.xlsx')
    ##############################
    
    
    
    
    
    ###USED TO MAKE ONE HOT ENCODING OF COUNTRY
    #df = one_hot_encode(df,c_name = 'Country')
    #oh_names = df.iloc[:,-25:].columns
    #index_oh = 2
    #for n in oh_names: # To move to position i'd like
    #    df.insert(index_oh,n,df.pop(n))
    #    index_oh+= 1
    #df.to_excel('bipbapboop4.xlsx')
    ##############################
    
    
    
    
    
    ###Used TO MAKE ONE HOT ENCODING OF SECTOR
    #df = one_hot_encode(df,c_name= 'Sector')
    #oh_names = df.iloc[:,-11:].columns
    #index_oh = 0
    #for n in oh_names: # To move to position i'd like
    #    df.insert(index_oh,n,df.pop(n))
    #    index_oh+= 1
    #df.to_excel('bipbapboop5.xlsx')
    #############################
    
    
    
    
    ###USED TO MAKE ONE HOT ENCODING OF INDUSTRY
    #df = one_hot_encode(df, c_name = 'Industry',sparsev=True)
    #oh_names = df.iloc[:,-125:].columns
    #index_oh = 11
    #for n in oh_names: # To move to position i'd like
    #    df.insert(index_oh,n,df.pop(n))
    #    index_oh+= 1
    #df.to_excel('PostOneHotStockData.xlsx')
    #############################
    #e.save_obj(df,'2YStockDFBcleaner')

    print('Donzo')
    f.close()
    
    
if __name__ == "__main__":
    main()
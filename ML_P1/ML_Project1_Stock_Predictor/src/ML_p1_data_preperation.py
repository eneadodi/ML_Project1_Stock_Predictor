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
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def filtered_columns(df):
    filter_vpw = [col for col in df if ' Volume' in col]
    filter_ppw = [col for col in df if ' Close' in col]
    time_rv = filter_vpw + filter_ppw
    filter_ntrv = [col for col in df if col not in time_rv]

    volume_per_week_df = df[filter_vpw]
    price_per_week_df = df[filter_ppw]
    categorical_values_df = df[filter_ntrv]
    return volume_per_week_df,price_per_week_df,categorical_values_df;

'''
Simple little helper function which creates a list of names for columns
given the start date and end date.
Interval is weekly.
THIS METHOD WAS CREATED AFTER pick_date_range
'''
def date_column_name_creator(startd,endd,extension="",date_as_string=True):
    start = datetime(startd[0],startd[1],startd[2])
    end = datetime(endd[0],endd[1],endd[2])
    l = []
    for dt in rrule.rrule(rrule.WEEKLY,dtstart=start,until=end):
        if date_as_string == True:
            l.append(str(dt.date()) + extension)
        else:
            l.append(dt.date())
    return l

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
    serv_outliers = serv[(serv >= serv_m + 1*serv_std)]
    serv_outliers.sort_values(ascending=False,inplace=True,na_position='first')
    
    serp_outliers = serp[(serp >= serp_m + 1*serp_std)]
    serp_outliers.sort_values(ascending=False,inplace=True,na_position='first')
    
    serc_outliers = serc[(serc >= serc_m + 1*serc_std)]
    serc_outliers.sort_values(ascending=False,inplace=True,na_position='first')

    size = str(len(df.index))
    
    f.write("VOLUME NULL OUTLIERS: \n")
    f.write(str(serv_outliers.shape[0]) + "/" + size + " are one standard deviation away from average null values per ticker\n")
    f.write(serv_outliers.to_string())
    f.write("\n\n\n")
    f.write("PRICE NULL OUTLIERS: \n")
    f.write(str(serp_outliers.shape[0]) +"/" + size + " are one standard deviation away from average null values per ticker\n")
    f.write(serp_outliers.to_string())
    f.write("\n\n\n")
    f.write("CATEGORICAL NULL OUTLIERS: \n")
    f.write(str(serc_outliers.shape[0]) + "/" + size + " are one standard deviation away from average null values per ticker\n")
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
        f.write('\n\n')
    
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
    pd.set_option('display.max_columns', 20)
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

def label_calc(x):
    if x >= 0.03:
        return 1
    else:
        return 0
    
def initalStart(dic_n):     
    #Load information
    ss = StockScraper()
    ss.scraped_info = e.load_obj(dic_n)
    stock_information_dict = ss.scraped_info
    ss.scraped_tickers = ss.extract_tickers()
    

    #check to see if each dictionary has same key values:
    check_identical_keys(stock_information_dict)
  
    
    #Create Pandas, reindex, and drop duplicates to support random order and avoid duplicates.
    df = pd.DataFrame(stock_information_dict, columns=stock_information_dict[0].keys())
    df.set_index('Ticker',inplace=True)
    df = df.reindex(np.random.permutation(df.index))
    #df.loc[df.astype(str).drop_duplicates().index] There are no duplicates :)
    
    
    
    #I have yet to decide if my Machine Learning Algorithm will be learning from a 2 year time span or a 5 year time span.
    # Thus I will make a copy of both options
    dstart2y = [2018,8,6]
    dstart5y = [2015,8,10]
    dend = [2020,8,14]
    for i in df.columns:
        print(i)

    df2y = pick_date_range(df,dstart2y,dend)
    df5y = pick_date_range(df, dstart5y, dend)
    df2y.to_pickle('2YStockDFLowCriteria',protocol=HIGHEST_PROTOCOL)
    df5y.to_pickle('5YStockDFBLowCriteria',protocol=HIGHEST_PROTOCOL)
    
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
                print_global_count() # To see if I miss a null or not.
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

'''
Stock prices are approximately log normal. Thus I decided on first transforming
stock price X -> Y = ln(x) then using the Standard scalar of sklearn.
'''
def lognormal_scale_rows(df):
    #First transform all values to log.
    for column in df:
        df.loc[:,column] = np.log(df[column])
    
    scaler = StandardScaler()
    df2 = pd.DataFrame(columns = df.columns,index=df.index,data=scaler.fit_transform(df.values.T).T)
    return df2

'''
It made sense to scale ticker Income and Sales based on the Sector group that they belong in.
This comes from the heuristic that different sectors have different ceilings.
'''
def scale_MinMax_by_Sector(df,s_or_i='Sales'):
    g = df.groupby(df['Sector'])
    df_section = df[[s_or_i]]
    names = g.groups.keys()
    scaler = MinMaxScaler()
    for n in names:

        dfn = g.get_group(n)[[s_or_i]]

        dfn_scaled = pd.DataFrame(index=dfn.index,data=scaler.fit_transform(dfn.values))
        
        df_section[df_section.index.isin(list(dfn.index))] = dfn_scaled

        dfn_scaled.to_excel('minmax'+n+'.xlsx')
    return df_section


'''
In case I just do not want to deal with any industry with less than 4 examples.
'''
def remove_bad_ticks_by_group(df,occurences,group_n):
    g = df.groupby(df[group_n]).count()
    
    #group_names = g.groups.keys()
    #print(g.columns)
    bad_g = list(g[g['Sector'] < occurences].index.values)
    print(bad_g)
    df_i = df[[group_n]]
    for index,row in df_i.iterrows():
        if row.values[0] in bad_g:
            df.drop(index,inplace=True)

    return df

''''Sector may be unnecessary because industry is a feature'''
def remove_category(df,substring):
    cols = [c for c in df.columns if substring not in c]
    rsdf = df[cols]
    rsdf.to_excel('rDF'+substring[:-1]+'.xlsx')
    return df


def remove_volume(df):
    cols = [c for c in df.columns if ' Volume' not in c]
    rvdf = df[cols]
    rvdf.to_excel('rvDF.xlsx')
    return rvdf 


        
        
''''A price/volume ratio that is lognormally scaled will likely be a better feature
than each individually. If needed to create lavel, final two price columns can optionally 
be saved and concatenated at the end.
Dates: 08-06-2018 -> 08-14-2020 
'''
def price_volume_ratio_feature_creator(df,startd,endd,leave_final_two_price_columns = True):
    
    #filters the volume and price DFs separately
    volume_per_week_df, price_per_week_df, categorical_values_df = filtered_columns(df)
    
    #to save if needed later.
    final_two_prices = price_per_week_df.iloc[:,-2:].copy()
    
    
    #to get column names
    col_names = date_column_name_creator(startd, endd, ' Price/Volume')
    
    
    #Create new feature which is the ratio of price and week.
    new_feat = price_per_week_df/volume_per_week_df.values[:,:]
    print('SIZE ,' , len(new_feat.columns))
    #print_full(new_feat)
    new_feat.columns = col_names
    
    new_feat.to_excel('ratio.xlsx')
    
    #lognormally scale new_feat
    new_feat = lognormal_scale_rows(new_feat)
     
    #create new DataFrame with price and volume replaced with the ratio price/volume
    pvrfdf = categorical_values_df.join(new_feat)

    if leave_final_two_price_columns == True:
        pvrfdf = pvrfdf.join(final_two_prices)
        
    pvrfdf.to_excel('ratiofit.xlsx')
    
    return pvrfdf
    
    
    
'''Maybe two years of Dates is unnecessary? If so, we can increase the number of examples.
   If the length of the Dates is not integer divisible by splits, then it will not include the last split.
   Splits parameter MUST be less than amount of dates.
   NOT TESTED
   ----------
'''
def split_dates_v_and_p(df,startd,endd,splits=4):
    
    
    #Generator to return list sections by size n
    def split(l,n):
        for i in range(0,len(l),n):
            yield l[i:i+n]
            
            
            
    #filters the volume and price DFs separately
    volume_per_week_df, price_per_week_df, categorical_values_df = filtered_columns(df)
    
    #Total amount of Date Columns
    total_cols = len(volume_per_week_df.columns)
    
    #Splitting dates will result in the column names needing to be changed.
    #Thus the new columns will be Date(xY) where Y is the Yth date plus the normal nondate columns
    counter = 1
    new_cols = list(categorical_values_df.columns)   
    for i in range(int((total_cols/splits)+0.99999)): #if perfect split, then don't add 1 to range.
        new_cols.append('Date(X'+str(counter)+') Price')
        new_cols.append('Date(X'+str(counter)+') Volume')

           
    
    sddf = pd.DataFrame(columns=new_cols)
    
    splitv = split(list(volume_per_week_df.columns),splits)
    splitp = split(list(price_per_week_df.columns),splits)
    
    for i in range(splits):
        df_v_split = df[next(splitv)]
        df_p_split = df[next(splitp)]
        df_split = df_v_split.join(df_p_split)
        df_split.reindex_axis(df_split.columns[::2].tolist() + df_split.columns[1::2].tolist(), axis=1)
        sddf_part = categorical_values_df.join(df_split)
        sddf = sddf.append(sddf_part)
    
    sddf.to_excel("splittingv1.xlsx")
    return sddf

'''
When the ratio dataframe is used, this method should be used rather than split_dates_v_and_p
'''
def split_dates(df,startd,endd,splits=4):
    #Generator to return list sections by size n
    def split(l,n):
        for i in range(0,len(l),n):
            yield l[i:i+n]
            
            
            
    #filters the volume and price DFs separately
    ilter_rpw = [col for col in df if ' Price/Volume' in col]
    filter_ntrv = [col for col in df if ' Price/Volume' not in col]

    ratio_per_week_df = df[ilter_rpw]
    categorical_values_df = df[filter_ntrv]
    
    
    #Total amount of Date Columns
    total_date_cols = len(ratio_per_week_df.columns)
    
    #Splitting dates will result in the column names needing to be changed.
    #Thus the new columns will be Date(xY) where Y is the Yth date plus the normal nondate columns
    counter = 1 # For Y value
    new_cols = list(categorical_values_df.columns)   
    new_cols_dates_length = int((total_date_cols/splits)+0.99999)
    
    for i in range(new_cols_dates_length): #if perfect split, then don't add 1 to range.
        new_cols.append('Date(X'+str(counter)+') Price/Volume')
        counter+=1
    
    '''Because we will likely have a final split with less columns than the preivous splits, then
    we need to append nan columns to the final split until they are equal in column lengths to use
    pd.concat on the list of dataframes. To do this I will need a new list that simply appends the lengths
    of each split.columns
    '''
    split_columns_length = [] 
    
    sddf = pd.DataFrame(columns=new_cols)
    split_dfs = []
    
    '''We need to track the final week Closing for each split to use as label. Thus we need to return 
    a list containing the final two column names of each split. We can use the date on these to calculate
    the labels
    '''
    final_two_each_split = []
    
    splitr = split(list(ratio_per_week_df.columns),new_cols_dates_length)
    handle_final_split = 1

    for i in range(splits):
        
        df_split = ratio_per_week_df[next(splitr)]
        final_two_each_split.append(df_split.iloc[:,-2:].columns)
        
        split_columns_length.append((len(df_split.columns) + len(categorical_values_df.columns)))
        sddf_part = categorical_values_df.join(df_split)
        
        if handle_final_split == splits: 
            iterations = split_columns_length[0] - split_columns_length[-1]
            print('made it here')
            for i in range(iterations):
                sddf_part['NAN COLUMNS ' + str(i)] = np.nan 
                
        sddf_part.columns = new_cols
        #sddf_part.to_excel('split'+str(handle_final_split)+'.xlsx')
        handle_final_split+=1
        split_dfs.append(sddf_part)
        
    
    sddf = pd.concat(split_dfs)
    #sddf.to_excel("splittingv1.xlsx")
    return final_two_each_split,sddf
    
def main():
    
    f = open('XXXXXX.txt','a')
    #####Will only be called once, when we have the dictionary but not pandas.
    #initalStart('FullStockDataLowCriteria')
    ############################
    
    
    
    
    df = pd.read_pickle('../data/2YStockDFRatio.pkl')
    
    #e.save_obj(df,'2YStockDFLowCriteria')
    #df5y = pd.read_pickle('5yStockDF')
    #df.to_excel('bipbapboop4.xlsx')
    
    
    #####Useful Filters
    volume_per_week_df, price_per_week_df, categorical_values_df = filtered_columns(df)
    ############################
    

    ####To get information on what values are still mising or NULL
    #ndf = categorical_values_df.isnull().sum(axis=0) #change categorical_values with volume_per_week or price_per_week if needed
    #print_full(ndf)
    ############################

    

    ####USED TO REMOVE WORST TICKERS (tickers with A LOT of missing features)
    #new_df = get_null_information('NullTickerInformationLowCriteria2YB', df,remove_worst=False)
    #new_df.to_pickle('2YStockDFBLowCriteriaclean.pkl',protocol=HIGHEST_PROTOCOL)
    #new_df.to_excel('2YStockDFBLowCriteriacleanexcel.xlsx')
    ############################
    
    
    
    
    ####USED TO REMOVE TICKERS WITH INDUSTRIES THAT APPEAR RARELY
    #df = remove_bad_ticks_by_group(df,5,'Industry')
    #e.save_obj(df,'2YStockDFLowCriteriaA')
    ############################
    
    
    
    ####USED TO FILL NULL INCOME VALUES WITH APPROPRIATE VALUES
    #fill_null_Income(df)
    #categorical_values_df.to_excel('boopdiboop.xlsx')
    #e.save_obj(df,'2YStockDFLowCriteriaB')
    #Pre method income avg: 769344444.4
    #Post method income avg: 783475326.4
    #############################
    
    
    
    ###Used to fill NULL SALES VALUES WITH APPROPRIATE VALUES 
    #fill_null_Sales(df)
    #categorical_values_df.to_excel('boopdibab.xlsx')
    #e.save_obj(df, '2YStockDFLowCriteriaB')
    #Pre method Sales avg: 11688020115
    #Post method Sales avg: 11762226708
    #############################
    

    
    
    ###USED TO SCALE INCOME AND SALES COLUMNS
    #df[['Income']] = scale_MinMax_by_Sector(df,'Income')
    #df[['Sales']] = scale_MinMax_by_Sector(df,'Sales')
    #df.to_excel('AllScaledData.xlsx')
    #e.save_obj(df,'2YStockDFLowCriteriaC')
    
    
    
    
    ###USED TO MAKE ONE HOT ENCODING OF RECOMMENDATOINS
    # Recommendations was a feature that was oringally in a Real Number scale from 1-5. 
    # However, the Integer number scale from 1-5 has a different meaning
    # Where: 1 -> Strong Buy
    #        2 -> Buy
    #        3 -> Hold 
    #        4 -> Sell 
    #        5 -> Strong Sell 
    #df = one_hot_encode(df,c_name = 'Recommendations',contains_null=True)
    #oh_names = df.iloc[:,-6:].columns #one hot names
    #index_oh = 5
    #for n in oh_names: # To move to position i'd like
    #   df.insert(index_oh,n,df.pop(n))
    #   index_oh+= 1
    #df.drop(columns='Recommendations_5.0',inplace=True) # If not 1-4, then must be 5, Also there is only one such
    #df.to_excel('bipbapboop3.xlsx')
    #e.save_obj(df,'2YStockDFLowCriteriaD')
    ##############################
    
    
    ###USED TO REMOVE TICKERS WHERE COUNTRY APPEARS VERY RARELY
    #df = remove_bad_ticks_by_group(df,5,'Country')
    #len_count = -1*len(df.groupby('Country').groups.keys())
    
    ###USED TO MAKE ONE HOT ENCODING OF COUNTRY
    #df = one_hot_encode(df,c_name = 'Country')
    #oh_names = df.iloc[:,len_count:].columns
    #index_oh = 2
    #for n in oh_names: # To move to position i'd like
    #    df.insert(index_oh,n,df.pop(n))
    #    index_oh+= 1
    #df.to_excel('bipbapboop4.xlsx')
    #e.save_obj(df,'2YStockDFLowCriteriaE')
    ##############################
    
    
    
    
    ###Used TO MAKE ONE HOT ENCODING OF SECTOR
    #df = one_hot_encode(df,c_name= 'Sector')
    #oh_names = df.iloc[:,-11:].columns
    #index_oh = 0
    #for n in oh_names: # To move to position i'd like
    #    df.insert(index_oh,n,df.pop(n))
    #    index_oh+= 1
    #df.to_excel('bipbapboop5.xlsx')
    #e.save_obj(df,'2YStockDFLowCriteriaF')
    #############################
    
    
    
    
    ###USED TO MAKE ONE HOT ENCODING OF INDUSTRY
    #df = one_hot_encode(df, c_name = 'Industry',sparsev=True)
    #oh_names = df.iloc[:,-130:].columns
    #index_oh = 11
    #for n in oh_names: # To move to position i'd like
    #    df.insert(index_oh,n,df.pop(n))
    #    index_oh+= 1
    #df.to_excel('PostOneHotStockData.xlsx')
    #e.save_obj(df,'2YStockDFLowCriteriaG')
    #############################



    ###USED TO SCALE PRICE AND VOLUME COLUMNS BY ROW    
    #print('telumpt')
    #df[filter_ppw] = log_normal_scale_rows(price_per_week_df)
    #df[filter_vpw] = log_normal_scale_rows(volume_per_week_df)
    #df.to_excel('PostScaling.xlsx')
    #e.save_obj(df,'2YStockDFLowCriteria')
    #############################
    
    
    
    
    
    ###USED TO MAKE LABEL COLUMN
    #df2 = pd.read_pickle('../data/PDPREONEHOT.pkl')
    #df2 = df2[df2.index.isin(list(df.index))]
    #
    #df2_c_w = df2['Date: 2020-08-03 Close']
    #df2_p_w = df2['Date: 2020-07-27 Close']
    #df2_i = df2_c_w - df2_p_w
    #print(df2_i)
    #df2_l = df2_i / df2_p_w
    #df2_l.to_frame()
    #print(df2_l)
    ##df2_l.to_excel('percentage_gain.xlsx')
    #df2_l = df2_l.apply(label_calc)
    ##df2_l.to_excel('percentage_gainBinary.xlsx')
    #del df['Date: 2020-08-03 Close']
    #df['Label'] = df2_l
    #df.to_excel('ReadyData.xlsx')
    #e.save_obj(df,'2YStockDFBcleaner')
    #############################
    
    
    
    
    
    ###USED TO MAKE DATASETS WITH REMOVED COLUMNS
    #se = remove_category(df2,'Sector_')
    #ind = remove_category(df2,'Industry_')
    #vol = remove_volume(df2)
    #
    #e.save_obj(se,'2YStockDFSR')
    #e.save_obj(ind,'2YStockDFIR')
    #e.save_obj(vol,'2YStockDFVR')
    #############################
    
    
    
    
    ###USED TO MAKE DATASET WITH PRICE/VOLUME RATIO FEATURE
    dstart2y = [2018,8,6]
    dend = [2020,8,10]
    #pdr = price_volume_ratio_feature_creator(df, dstart2y, dend)
    #e.save_obj(pdr,'2YStockDFRatio')
    #############################
    
    
    
    
    
    ###USED TO MAKE ALTERNATE DATAFRAME WITH LESS COLUMNS AND MORE ROWS. Removes all non Dated values too.
    #df = df.drop(labels = ['Date: 2020-08-03 Close','Date: 2020-08-10 Close'],axis=1)
    #sdf_label_cols, sdf = split_dates(df, dstart2y, dend, 4)
    #filter_ntrv = [col for col in sdf if ' Price/Volume' not in col]
    #sdf.drop(df[filter_ntrv],axis=1,inplace=True)
    #print('making excel')
    #sdf.to_excel('splitRatioOnly.xlsx')
    #e.save_obj(sdf,'2YStockDFRatioSplit')
    #e.save_obj(sdf_label_cols,'2YStockDFRatioSplitLabelColumns')
    #############################
    
    
    
    print('Donzo')
    
    f.close()
    

    
    
if __name__ == "__main__":
    main()
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
import copy
import collections
from pandas.tests.frame.test_sort_values_level_as_str import ascending
from dateutil import rrule
from datetime import datetime, timedelta
import yfinance as yf
import ML_p1_data_preperation as p
from yfinance.utils import auto_adjust
#from sympy import S, symbols, printing

    


   

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
    SVM_process1()
    
if __name__ == '__main__':
    main()
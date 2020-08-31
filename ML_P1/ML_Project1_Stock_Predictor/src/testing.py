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

    
def LSTM_process():
    #df = pd.read_pickle('../data/2YStockDFLowCriteria.pkl')
    #ticker_names = prepare_LSTM_data(df)
    #e.save_obj(ticker_names,'ticker_names')
    
    LSTM_data_X = e.load_obj('../data/x_ticker_time_series_low_criteria')
    LSTM_data_Y = fix_y_values(e.load_obj('../data/y_ticker_time_series_low_criteria'))
    ticker_names = e.load_obj('ticker_names')
    
    x_train = LSTM_data_X[0:3041]
    x_validate = LSTM_data_X[2366:3041]
    x_test = LSTM_data_X[3041:]
    
    y_train = LSTM_data_Y[0:3041]
    y_validate = LSTM_data_Y[2366:3041]
    y_test = LSTM_data_Y[3041:]
    
    ticker_names = ticker_names[3041:]
    
    x_train = numpify_LSTM_data(x_train)
    y_train = numpify_LSTM_data(y_train)
    print('ready to create model')
    model = create_LSTM_model(0.002,x_train)
    print('ready to train model')
    epochs,mse,history = train_LSTM_model(model, x_train, y_train, 1, 20,v_split=0.2)
    print('model trained.')
    list_of_metrics_to_plot = ['accuracy'] 
    plot_the_loss_curve(epochs, mse)

    compare_LSTM_results(model,x_test,y_test,ticker_names)
    
    
'''Maybe two years of Dates is unnecessary? If so, we can increase the number of examples.
   If the length of the Dates is not integer divisible by splits, then it will not include the last split.
   Splits parameter MUST be less than amount of dates.
'''
def split_dates_v_and_p(df,startd,endd,splits=4):
    
    
    #Generator to return list sections by size n
    def split(l,n):
        for i in range(0,len(l),n):
            yield l[i:i+n]
            
            
            
    #filters the volume and price DFs separately
    volume_per_week_df, price_per_week_df, categorical_values_df = p.filtered_columns(df)
    
    #Total amount of Date Columns
    total_date_cols = len(volume_per_week_df.columns)
    
    #Splitting dates will result in the column names needing to be changed.
    #Thus the new columns will be Date(xY) where Y is the Yth date plus the normal nondate columns
    counter = 1
    new_cols = list(categorical_values_df.columns)   
    new_cols_dates_length = int((total_date_cols/splits)+0.99999)

    for i in range(new_cols_dates_length): #if perfect split, then don't add 1 to range.

        new_cols.append('Date(X'+str(counter)+') Volume')
        counter += 1
    
    counter = 1
    for i in range(new_cols_dates_length): #if perfect split, then don't add 1 to range.
        new_cols.append('Date(X'+str(counter)+') Close')
        counter += 1
            
    '''Because we will likely have a final split with less columns than the preivous splits, then
    we need to append nan columns to the final split until they are equal in column lengths to use
    pd.concat on the list of dataframes. To do this I will need a new list that simply appends the lengths
    of each split.columns
    '''
    split_columns_length = [] 
    
    sddf = pd.DataFrame(columns=new_cols)
    split_dfs = []
    final_two_each_split = []
    
    splitr = split(list(price_per_week_df.columns),new_cols_dates_length)
    handle_final_split = 1
    splitr2 = split(list(volume_per_week_df.columns),new_cols_dates_length)
    
    for i in range(splits):
        print(i)
        df_split1 = price_per_week_df[next(splitr)]
        df_split2 = volume_per_week_df[next(splitr2)]
        final_two_each_split.append(df_split1.iloc[:,-2:].columns)
        
        split_columns_length.append((len(df_split1.columns) + len(categorical_values_df.columns) + len(df_split2.columns)))
        sddf_part = categorical_values_df.join(df_split2)
        sddf_part = sddf_part.join(df_split1)
        
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

def SVM_process1():
    '''For the first process I will be using the df with removed industry.
        I am going to need to import two dataframes.
            The first will be one without price scaling.This'll be used purely for
        label creation.
            
            The second will be a ratio dataframe where values are 
        
        
    '''
    '''
    df = pd.read_pickle('../data/Low_Criteria_Pre_price_v_scale.pkl')
    dstart2y = [2018,8,6]
    dend = [2020,8,10]

    final_two_split, df2 = split_dates_v_and_p(df,dstart2y,dend,5)
    
    df2 = df2.iloc[:13580]
    e.save_obj(df2,'splitpriceright')
    
    #now we split price and volume separately and remove last two columns of each split df

    vpw,ppw,cw = p.filtered_columns(df2)
    
    label_price = copy.deepcopy(ppw[ppw.columns[-2:]])
    ppw = ppw[ppw.columns[:-1]]
    vpw = vpw[vpw.columns[:-1]]

    #now we combine the columns again.
    df2 = pd.concat([cw,vpw,ppw],axis=1)


    #we make the label column and join it to df2.
    label_price['Gain %'] =  label_price.apply(lambda row: ((row.iloc[1] - row.iloc[0])/np.abs(row.iloc[0]))*100,axis=1)
    label_price['label'] = label_price.apply(lambda row: (1 if (row.loc['Gain %'] >= 3) else 0),axis=1)

    l_col = label_price['label']
    print(l_col.shape)
    print(df2.shape)
    df2 = pd.concat([df2,l_col],axis =1)
    e.save_obj(df2,'second_check_point')
    
    df = pd.read_pickle('second_check_point.pkl')
    df.to_excel('sffs.xlsx')'''
    print('done')
def main():
    SVM_process1()
    
if __name__ == '__main__':
    main()
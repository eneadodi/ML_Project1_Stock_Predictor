import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn
import yfinance as yf
from click._compat import iteritems
import requests
import time
import Web_Stock_Scrapper



data = yf.download(tickers = 'ATHE', period = '5y', interval='1wk', auto_adjust=True)
print(data.columns)
spy = yf.Ticker("IRTC")
spy_i = spy.info
for key, value in spy_i.items():
    print(key)

#book value, forwardEPS, averageVolume (3 month), threeYearAverageReturn ,Two Hundred Day Average
print('navPrice:')
print(spy_i['navPrice'])
print('regularMarketOpen:')
print(spy_i['regularMarketOpen'])
print('TwoHundredDayAverage:')
print(spy_i['twoHundredDayAverage'])
print('payoutRatio:')
print(spy_i['payoutRatio'])
print('priceHint:')
print(spy_i['priceHint'])
print('regularMarketVolume:')
print(spy_i['regularMarketVolume'])
print('averageVolume:')
print(spy_i['averageVolume'])
print('bookValue:')
print(spy_i['bookValue'])
print('shortRatio:')
print(spy_i['shortRatio'])
print('threeYearAverageReturn:')
print(spy_i['threeYearAverageReturn'])
print('forwardEPS:')
print(spy_i['forwardEps'])
print("dividends")
print(spy.dividends)
print("financials")
print(spy.sustainability)
print("Institutional holders")
print(spy._institutional_holders)


'''
spy_m = spy.major_holders
for i in spy_m:
    print("major holder: " + i)
    
spy_ih = spy.institutional_holders

print(spy_ih.head(10))
'''
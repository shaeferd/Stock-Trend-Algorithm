
# coding: utf-8

# In[1]:

import pandas as pd
#from pandas_datareader import data
import datetime
#get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import fix_yahoo_finance as yf
from pytrends.request import TrendReq
import numpy as np
#import newsapia
#from newsapi.articles import Articles
#from newsapi import NewsApiClient
import requests
from pytrends.request import TrendReq
import numpy as np
import statsmodels.api as sm
import collections
from itertools import count
from collections import OrderedDict
from scipy import stats
import seaborn as sns
# from pandas.tools.plotting import table
#from sklearn.model_selection import cross_val_score
#import sklearn.metrics as skmetric


pd.options.mode.chained_assignment = None
import math
import sqlite3

###TODO:
### - Redefine crossed_line so you can just test 1 date and backtest, and add actual price diff
### backtest idea: add all sells up, subtract from buys and neutralizes (this only counts shorts)... Can do same other way around and add 2 numbers together
### if you do idea above will either have to change if consecutive buys or account for number of shares you must neutralize
### Add new script where you just test the current day/week

# #### Notes: Good to run. Now can see buy only profit. Recommend using the specific start date to set testing threshold when making purchases. + recommend backtesting at least once for each stock.

# ## Stock Data

# ### Pull from diff dates every time 
# #### - just pull from these dates first and resample and get number of rows, then split and pull from those dates in the split

# In[19]:


stock = 'IMGN'
start = '2014-12-18'
end = '2019-12-17'#10-10 #11-26

table_name = 'stock_'+stock+start+end
table_name = table_name.replace('-', '_')

print(table_name)

con = sqlite3.connect("Reco.db")
cur = con.cursor()

try:
    amd = pd.read_sql_query('SELECT * FROM '+table_name+';', con = con, index_col = 'Date')
    print('read from cache')
except:
    amd = yf.download(stock, start, end)
    amd.to_sql(table_name, con = con)
    print('cached')
#amd = yf.download(stock, start, end)

amd = amd.Open
amd.index = pd.to_datetime(amd.index)

weekly_amd = pd.DataFrame(amd.resample('W').first().shift(periods = -1, freq = 'W'))

# new_price = pd.Series({'Open' : 160.87})
# new_price.name = datetime.datetime(2019,5,12)
# weekly_amd = weekly_amd.append(new_price)


# ## Google Trends Data

# #### Why does relative trend score fuck it up?
# #### 1. not huge problem, but normalizing twice
# #### 2. recommendations change for shifts in time ranges (because "num_searches" changes for train and test data)
# ##### Must shift time ranges in order to test present data
# #### 3. Since test recommendation isn't same as real recommendation, can't label it as an accurate test
# #### Solutions:
# #### - re-collect the trands data for every single testing datapoint
# 
# #### TODO:
# #### - Figure out way to split between train and test (just mark a date) but then fit train to everything before the last value and test the last value.
# #### - have buy and sell values populated after each run (maybe make new dataframe, appending each answer column to train, where applicable
table_name_trend = 'trend_'+stock+start+end
table_name_trend = table_name_trend.replace('-', '_')
try:
    amd_trend = pd.read_sql_query('SELECT * FROM '+table_name_trend+';', con = con, index_col = 'date')
    amd_trend.index = pd.to_datetime(amd_trend.index)
    print('read from cache')
except:
    pytrends = TrendReq(hl='en-US', tz=360)
    kw_list = [stock] #can try with neg and pos phrases like "buy amd", "sell amd"
    pytrends.build_payload(kw_list, cat = 7, timeframe=start + ' ' + end)
    amd_trend = pytrends.interest_over_time()
    amd_trend = amd_trend[[stock]]
    amd_trend.to_sql(table_name_trend, con = con)
    amd_trend.index = pd.to_datetime(amd_trend.index)
    #cur.execute('CREATE TABLE '+table_name_trend+' (date STRING, price NUMBER, reco TEXT, simple_reco TEXT)')
    print('cached')

# new_trend = pd.Series({stock : 0})
# new_trend.name = datetime.datetime(2019,5,12)
# amd_trend = amd_trend.append(new_trend)

amd_trend = amd_trend.shift(periods = 1, freq = 'W')
print(amd_trend.tail())
combined = weekly_amd.merge(amd_trend, left_index = True, right_index = True, how = 'outer')
print(combined.tail())
df_amd = combined.rename(columns = {'Open': 'price', stock: 'num_searches'})
df_amd = df_amd.dropna() #only drops first and last date which don't match up
# df_amd.head()

rolling_price = df_amd['price'].rolling(window = 5, center = True, min_periods = 1)
rolling_trend = df_amd['num_searches'].rolling(window = 5, center = True, min_periods = 1)
df_amd['rolling_price'] = rolling_price.mean()
df_amd['rolling_trend'] = rolling_trend.mean()
# df_amd.head()


###KEEP below

#df_amd.loc['2019-05-5', 'price'] = 174.66

#df_amd.loc['2019-01-27', 'num_searches'] = 30


# ## Cross correlation

# In[528]:

# ccf = sm.tsa.stattools.ccf(df_amd['num_searches'], df_amd['price'] )

# plt.plot(ccf)
# plt.xlim(-30,50)
# plt.ylim(-2, 0.9)
# plt.axhline(0, color="black", linewidth=1)
# plt.title('Cross Correlation: "num_searches" vs. "price"')
# plt.xlabel('lag')
# plt.ylabel('correlation')



# ### cross correlation confirms no lag

# In[530]:

print(np.corrcoef(df_amd['num_searches'], df_amd['price']))


# #### High correlation between number of searches and price for this time interval

# In[531]:

# df_amd.plot()


# In[532]:

plt.plot(df_amd[['price', 'rolling_price']])
plt.title('Price and Rolling Price over time for '+stock)
plt.xlabel('Date')
plt.ylabel('Price')


# # In[533]:

plt.plot(df_amd[['num_searches', 'rolling_trend']])
plt.title('Trend and Rolling Trend over time for '+stock)
plt.xlabel('Date')
plt.ylabel('Number of Searches')


# In[534]:

df_amd['trend_diff'] = df_amd['num_searches'] - df_amd['rolling_trend']
df_amd['price_diff'] = df_amd['price'] - df_amd['rolling_price']

# threshold = int(.7 * len(df_amd))
# df_train_whole = df_amd[:threshold]
# df_test_whole = df_amd[threshold:]
df_train_whole = df_amd[:-1]
df_test_whole = df_amd[-1:]
reco_list = []



cur.execute('DROP TABLE IF EXISTS '+stock)
cur.execute('CREATE TABLE '+stock+' (date STRING, price NUMBER, reco TEXT, simple_reco TEXT)')

for i in range(0, len(df_test_whole)-1):


    start = df_train_whole.iloc[0].name.strftime('%Y-%m-%d')
    # print('start', start)
    end = df_test_whole.iloc[i].name+datetime.timedelta(days=7)
    end = end.strftime('%Y-%m-%d')
    # print('end', end)
    # print('start', type(start))
    table_name = 'stock_'+stock+start+end
    table_name = table_name.replace('-', '_')
    try:
        amd = pd.read_sql_query('SELECT * FROM '+table_name+';', con = con, index_col = 'Date')
        print('read from cache')
    except:
        amd = yf.download(stock, start, end)
        amd.to_sql(table_name, con = con)
        print('cached')

    amd = amd.Open

    amd.index = pd.to_datetime(amd.index)

    weekly_amd = pd.DataFrame(amd.resample('W').first().shift(periods = -1, freq = 'W'))

    table_name_trend = 'trend_'+stock+start+end
    table_name_trend = table_name_trend.replace('-', '_')
    try:
        amd_trend = pd.read_sql_query('SELECT * FROM '+table_name_trend+';', con = con, index_col = 'date')
        amd_trend.index = pd.to_datetime(amd_trend.index)
        print('read from cache')
    except:
        pytrends = TrendReq(hl='en-US', tz=360, timeout = 10)
        kw_list = [stock] #can try with neg and pos phrases like "buy amd", "sell amd"
        pytrends.build_payload(kw_list, cat = 7, timeframe=start + ' ' + end)
        amd_trend = pytrends.interest_over_time()
        amd_trend = amd_trend[[stock]]
        amd_trend.to_sql(table_name_trend, con = con)
        amd_trend.index = pd.to_datetime(amd_trend.index)
        #cur.execute('CREATE TABLE '+table_name_trend+' (date STRING, price NUMBER, reco TEXT, simple_reco TEXT)')
        print('cached')

    # pytrends = TrendReq(hl='en-US', tz=360, timeout = 10)



    # kw_list = [stock] #can try with neg and pos phrases like "buy amd", "sell amd"
    # pytrends.build_payload(kw_list, cat = 7, timeframe=start + ' ' + end)



    # amd_trend = pytrends.interest_over_time()
    # amd_trend = amd_trend[[stock]]

    # new_trend = pd.Series({stock : 0})
    # new_trend.name = datetime.datetime(2019,5,12)
    # amd_trend = amd_trend.append(new_trend)

    amd_trend = amd_trend.shift(periods = 1, freq = 'W')
    combined = weekly_amd.merge(amd_trend, left_index = True, right_index = True, how = 'outer')

    df_amd = combined.rename(columns = {'Open': 'price', stock: 'num_searches'})
    df_amd = df_amd.dropna() #only drops first and last date which don't match up
    # df_amd.head()
    #Change final price here
    #df_amd.loc['2019-05-5', 'price'] = 174.66

    rolling_price = df_amd['price'].rolling(window = 5, center = True, min_periods = 1)
    rolling_trend = df_amd['num_searches'].rolling(window = 5, center = True, min_periods = 1)
    df_amd['rolling_price'] = rolling_price.mean()
    df_amd['rolling_trend'] = rolling_trend.mean()

    #rt_diff = [0]
    td_diff = [0]
    for i in range(len(df_amd)-1):
        td_diff.append(df_amd.rolling_trend.iloc[i+1] - df_amd.rolling_trend.iloc[i])
    df_amd['td_diff'] = td_diff

    #rp_diff = [0]
    pd_diff = [0]
    for i in range(len(df_amd)-1):
        pd_diff.append(df_amd.rolling_price.iloc[i+1] - df_amd.rolling_price.iloc[i])
    df_amd['pd_diff'] = pd_diff


    # df_amd.head()
    #ccf = sm.tsa.stattools.ccf(df_amd['num_searches'], df_amd['price'] )
    print('trend v price correlation')
    print(np.corrcoef(df_amd['num_searches'], df_amd['price']))
    print('rolling trend v price correlation')
    print(np.corrcoef(df_amd['rolling_trend'], df_amd['rolling_price']))
    df_amd['trend_diff'] = df_amd['num_searches'] - df_amd['rolling_trend']
    df_amd['price_diff'] = df_amd['price'] - df_amd['rolling_price']

    td_diff = [0]
    for i in range(len(df_amd)-1):
        td_diff.append(df_amd.num_searches.iloc[i+1] - df_amd.num_searches.iloc[i])
    df_amd['td_diff'] = td_diff

    pd_diff = [0]
    for i in range(len(df_amd)-1):
        pd_diff.append(df_amd.price.iloc[i+1] - df_amd.price.iloc[i])
    df_amd['pd_diff'] = pd_diff



    print('trend v price diff')
    print(np.corrcoef(df_amd['trend_diff'], df_amd['price_diff']))

    # ccf = sm.tsa.stattools.ccf(df_amd['trend_diff'], df_amd['price_diff'] )
    # print(ccf[0:5])


    print('trend v price diff derivative')
    print(np.corrcoef(df_amd['td_diff'], df_amd['pd_diff']))
    ccf = sm.tsa.stattools.ccf(df_amd['td_diff'], df_amd['pd_diff'] )
    print(ccf[0:5])

    #print(df_amd.tail())
    #label train and test split conditional on i

    df_train = df_amd.iloc[:-1]
    df_test = df_amd.iloc[-1:]

    df_train_norm = df_train.copy()
    #get z-score
    trend_diff_mean = np.mean(df_train_norm['trend_diff'])#.mean()
    price_diff_mean = np.mean(df_train_norm['price_diff'])#.mean()

    trend_diff_std = np.std(df_train_norm['trend_diff'])
    price_diff_std = np.std(df_train_norm['price_diff'])

    # df_train_norm.iloc[:,4:6] = df_train_norm.apply(lambda x: (x-np.mean(x))/np.std(x))
    df_train_norm['trend_diff'] = (df_train_norm['trend_diff'] - trend_diff_mean) / trend_diff_std
    df_train_norm['price_diff'] = (df_train_norm['price_diff'] - price_diff_mean) / price_diff_std
    print('normalized trend v price diff')
    print(np.corrcoef(df_train_norm['trend_diff'], df_train_norm['price_diff']))
    ccf = sm.tsa.stattools.ccf(df_amd['trend_diff'], df_amd['price_diff'] )
    print(ccf[0:5])

    #derivative setup
    td_diff_mean = np.mean(df_train_norm['td_diff'])#.mean()
    pd_diff_mean = np.mean(df_train_norm['pd_diff'])#.mean()

    td_diff_std = np.std(df_train_norm['td_diff'])
    pd_diff_std = np.std(df_train_norm['pd_diff'])

    # df_train_norm.iloc[:,4:6] = df_train_norm.apply(lambda x: (x-np.mean(x))/np.std(x))
    df_train_norm['td_diff'] = (df_train_norm['td_diff'] - td_diff_mean) / td_diff_std
    df_train_norm['pd_diff'] = (df_train_norm['pd_diff'] - pd_diff_mean) / pd_diff_std
    print('normalized trend v price derviative diff')
    print(np.corrcoef(df_train_norm['td_diff'], df_train_norm['pd_diff']))



    #derivative setup
    td_diff_mean = np.mean(df_train_norm['td_diff'])#.mean()
    pd_diff_mean = np.mean(df_train_norm['pd_diff'])#.mean()

    td_diff_std = np.std(df_train_norm['td_diff'])
    pd_diff_std = np.std(df_train_norm['pd_diff'])

    # df_train_norm.iloc[:,4:6] = df_train_norm.apply(lambda x: (x-np.mean(x))/np.std(x))
    df_train_norm['td_diff'] = (df_train_norm['td_diff'] - td_diff_mean) / td_diff_std
    df_train_norm['pd_diff'] = (df_train_norm['pd_diff'] - pd_diff_mean) / pd_diff_std
    print('normalized trend v price derviative diff')
    print(np.corrcoef(df_train_norm['td_diff'], df_train_norm['pd_diff']))



    # ccf = sm.tsa.stattools.ccf(df_amd['td_diff'], df_amd['pd_diff'] )
    # print(ccf[0:5])

    #Extract interval ranges from training data
    '''
    pd_q75, pd_q60, pd_q40, pd_q25 = np.percentile(df_train_norm['price_diff'], [75, 60, 40, 25])
    td_q75, td_q60, td_q40, td_q25 = np.percentile(df_train_norm['trend_diff'], [75, 60, 40, 25])
    '''

    #derivative
    pd_q75, pd_q60, pd_q40, pd_q25 = np.percentile(df_train_norm['pd_diff'], [99.9, 60, 40, .1])
    td_q75, td_q60, td_q40, td_q25 = np.percentile(df_train_norm['td_diff'], [99.9, 60, 40, .1])

    def crossed_line(df):
        cross_lst = [False]
        for i in range(len(df['price_diff']) - 1):
            if(((df['price_diff'][i] < 0) & (df['price_diff'][i+1] > 0)) |\
            ((df['price_diff'][i] > 0) & (df['price_diff'][i+1] < 0))):
                cross_lst.append(True)
            else:
                cross_lst.append(False)
        return cross_lst
    MY_RISK = 10
    BASE_RISK = 10
    MY_PORTFOLIO = 1000

    #if price diff is greater than 75% interval and trend diff is less than 60%, short.
    #if price diff is less than 25% interval and trend diff greater than 40%, buy.
    #if trend diff is greater than 75% interval and price diff is less than 60%, buy
    #if trend diff is less than 25% interval and price diff is greater than 40%, short
    #if we have positions and the price crosses the rolling price (goes from pos to neg or neg to pos price diff),
    #close positions
    #else: hold
    num_positions = 0
    num_short = 0
    num_long = 0
    #price_diff and trend_diff aren't correlated, keep that in mind
    #could change this logic to bet on volatility
    '''
    def simple_reco(df):#change these to 40 and 60 in next test
        if((df['price_diff'] > pd_q75) & (df['trend_diff'] < td_q60) & (df['trend_diff'] > td_q25)):
            return 'sell'
        elif((df['price_diff'] < pd_q25) & (df['trend_diff'] > td_q40) & (df['trend_diff'] < td_q75)):
            return 'buy'
        elif((df['trend_diff'] > td_q75) & (df['price_diff'] < pd_q60) & (df['price_diff'] > td_q25)):#can search for wild positive trend change, but this doesn't necessarily mean buy, can also be negative news
            return 'buy'
        elif((df['trend_diff'] < td_q25) & (df['price_diff'] > pd_q40) & (df['price_diff'] < td_q75)):
            return 'sell'
        # elif((df['crossed'] == True)):
        #     return 'neutralize'
        else:
            return 'neutralize'
    '''
    #deriv
    def simple_reco(df):#change these to 40 and 60 in next test
        # if((df['pd_diff'] > pd_q75) & (df['td_diff'] < td_q60) & (df['td_diff'] > td_q25)):
        #     return 'sell'
        if((df['pd_diff'] < pd_q25) & (df['td_diff'] > td_q40) & (df['td_diff'] < td_q75)):
            return 'buy'
        elif((df['td_diff'] > td_q75) & (df['pd_diff'] < pd_q60) & (df['pd_diff'] > td_q25)):#can search for wild positive trend change, but this doesn't necessarily mean buy, can also be negative news
            return 'buy'
        # elif((df['td_diff'] < td_q25) & (df['pd_diff'] > pd_q40) & (df['pd_diff'] < td_q75)):
        #     return 'sell'
        # elif((df['crossed'] == True)):
        #     return 'neutralize'
        else:
            return 'neutralize'


    def reco(df, risk = MY_RISK):
        if df['td_diff'] > 0:
            return 'buy'
        elif df['td_diff'] < 0:
            return 'sell'
        else:
            return 'hold'
    

    ### Normalize price and trend differences in test data
    #get z-score
    df_test_norm = df_test.copy()

    df_test_norm['td_diff'] = (df_test_norm['td_diff'] - td_diff_mean) / td_diff_std
    df_test_norm['pd_diff'] = (df_test_norm['pd_diff'] - pd_diff_mean) / pd_diff_std

    #populating values for train
    df_norm = df_train_norm.append(df_test_norm)
    #df_test = df_amd.iloc[-1:]

    #crossed
    
    df_norm['crossed'] = crossed_line(df_norm)

    df_test_norm = df_norm.iloc[-1:]
    df_train_norm = df_norm.iloc[:-1]
    

    # df_test_norm['crossed'] = crossed_line(df_test_norm)

    df_test_norm['recommendation'] = df_test_norm.apply(reco,risk = MY_RISK, axis = 1)
    df_test_norm['simple_reco'] = df_test_norm.apply(simple_reco, axis = 1)
    # print(df_test_norm)
    tup_1 = (df_test_norm.iloc[0].name.strftime('%Y-%m-%d'), df_test_norm['price'][0], df_test_norm['recommendation'][0], df_test_norm['simple_reco'][0])
    #tup_2 = (df_test_norm.iloc[1].name.strftime('%Y-%m-%d'), df_test_norm['price'][1], df_test_norm['recommendation'][1], df_test_norm['simple_reco'][1])
    #cur.execute('DELETE FROM '+stock' WHERE name = ?', (fighter,))
    cur.execute('INSERT INTO '+stock+' VALUES (?, ?, ?, ?)', tup_1)
    #cur.execute('INSERT INTO '+stock+' VALUES (?, ?, ?, ?)', tup_2)
    con.commit()




#Single case: delete 3 quotes below

i = 0
start = df_train_whole.iloc[0].name.strftime('%Y-%m-%d')
print('start', start)
end = df_test_whole.iloc[i].name+datetime.timedelta(days=7)
end = end.strftime('%Y-%m-%d')
print('end', end)
# print('start', type(start))
table_name = 'stock_'+stock+start+end
table_name = table_name.replace('-', '_')
try:
    amd = pd.read_sql_query('SELECT * FROM '+table_name+';', con = con, index_col = 'Date')
    print('read from cache')
except:
    amd = yf.download(stock, start, end)
    amd.to_sql(table_name, con = con)
    print('cached')

amd = amd.Open

amd.index = pd.to_datetime(amd.index)

weekly_amd = pd.DataFrame(amd.resample('W').first().shift(periods = -1, freq = 'W'))

table_name_trend = 'trend_'+stock+start+end
table_name_trend = table_name_trend.replace('-', '_')
try:
    amd_trend = pd.read_sql_query('SELECT * FROM '+table_name_trend+';', con = con, index_col = 'date')
    amd_trend.index = pd.to_datetime(amd_trend.index)
    print('read from cache')
except:
    pytrends = TrendReq(hl='en-US', tz=360)
    kw_list = [stock] #can try with neg and pos phrases like "buy amd", "sell amd"
    pytrends.build_payload(kw_list, cat = 7, timeframe=start + ' ' + end)
    amd_trend = pytrends.interest_over_time()
    amd_trend = amd_trend[[stock]]
    amd_trend.to_sql(table_name_trend, con = con)
    amd_trend.index = pd.to_datetime(amd_trend.index)
    #cur.execute('CREATE TABLE '+table_name_trend+' (date STRING, price NUMBER, reco TEXT, simple_reco TEXT)')
    print('cached')

pytrends = TrendReq(hl='en-US', tz=360, timeout = 10)



kw_list = [stock] #can try with neg and pos phrases like "buy amd", "sell amd"
pytrends.build_payload(kw_list, cat = 7, timeframe=start + ' ' + end)



amd_trend = pytrends.interest_over_time()
amd_trend = amd_trend[[stock]]

# new_trend = pd.Series({stock : 0})
# new_trend.name = datetime.datetime(2019,5,12)
# amd_trend = amd_trend.append(new_trend)

amd_trend = amd_trend.shift(periods = 1, freq = 'W')
combined = weekly_amd.merge(amd_trend, left_index = True, right_index = True, how = 'outer')

df_amd = combined.rename(columns = {'Open': 'price', stock: 'num_searches'})
df_amd = df_amd.dropna() #only drops first and last date which don't match up
# df_amd.head()
#Change final price here
print(df_amd)
df_amd.loc['2019-12-09', 'price'] = 4.12

rolling_price = df_amd['price'].rolling(window = 5, center = True, min_periods = 1)
rolling_trend = df_amd['num_searches'].rolling(window = 5, center = True, min_periods = 1)
df_amd['rolling_price'] = rolling_price.mean()
df_amd['rolling_trend'] = rolling_trend.mean()

#rt_diff = [0]
td_diff = [0]
for i in range(len(df_amd)-1):
    td_diff.append(df_amd.rolling_trend.iloc[i+1] - df_amd.rolling_trend.iloc[i])
df_amd['td_diff'] = td_diff

#rp_diff = [0]
pd_diff = [0]
for i in range(len(df_amd)-1):
    pd_diff.append(df_amd.rolling_price.iloc[i+1] - df_amd.rolling_price.iloc[i])
df_amd['pd_diff'] = pd_diff


# df_amd.head()
#ccf = sm.tsa.stattools.ccf(df_amd['num_searches'], df_amd['price'] )
print('trend v price correlation')
print(np.corrcoef(df_amd['num_searches'], df_amd['price']))
print('rolling trend v price correlation')
print(np.corrcoef(df_amd['rolling_trend'], df_amd['rolling_price']))
df_amd['trend_diff'] = df_amd['num_searches'] - df_amd['rolling_trend']
df_amd['price_diff'] = df_amd['price'] - df_amd['rolling_price']

td_diff = [0]
for i in range(len(df_amd)-1):
    td_diff.append(df_amd.trend_diff.iloc[i+1] - df_amd.trend_diff.iloc[i])
df_amd['td_diff'] = td_diff

pd_diff = [0]
for i in range(len(df_amd)-1):
    pd_diff.append(df_amd.price_diff.iloc[i+1] - df_amd.price_diff.iloc[i])
df_amd['pd_diff'] = pd_diff



print('trend v price diff')
print(np.corrcoef(df_amd['trend_diff'], df_amd['price_diff']))

# ccf = sm.tsa.stattools.ccf(df_amd['trend_diff'], df_amd['price_diff'] )
# print(ccf[0:5])


print('trend v price diff derivative')
print(np.corrcoef(df_amd['td_diff'], df_amd['pd_diff']))
ccf = sm.tsa.stattools.ccf(df_amd['td_diff'], df_amd['pd_diff'] )
print(ccf[0:5])

#print(df_amd.tail())
#label train and test split conditional on i

df_train = df_amd.iloc[:-1]
df_test = df_amd.iloc[-1:]

df_train_norm = df_train.copy()
#get z-score
trend_diff_mean = np.mean(df_train_norm['trend_diff'])#.mean()
price_diff_mean = np.mean(df_train_norm['price_diff'])#.mean()

trend_diff_std = np.std(df_train_norm['trend_diff'])
price_diff_std = np.std(df_train_norm['price_diff'])

# df_train_norm.iloc[:,4:6] = df_train_norm.apply(lambda x: (x-np.mean(x))/np.std(x))
df_train_norm['trend_diff'] = (df_train_norm['trend_diff'] - trend_diff_mean) / trend_diff_std
df_train_norm['price_diff'] = (df_train_norm['price_diff'] - price_diff_mean) / price_diff_std
print('normalized trend v price diff')
print(np.corrcoef(df_train_norm['trend_diff'], df_train_norm['price_diff']))
ccf = sm.tsa.stattools.ccf(df_amd['trend_diff'], df_amd['price_diff'] )
print(ccf[0:5])

#derivative setup
td_diff_mean = np.mean(df_train_norm['td_diff'])#.mean()
pd_diff_mean = np.mean(df_train_norm['pd_diff'])#.mean()

td_diff_std = np.std(df_train_norm['td_diff'])
pd_diff_std = np.std(df_train_norm['pd_diff'])

# df_train_norm.iloc[:,4:6] = df_train_norm.apply(lambda x: (x-np.mean(x))/np.std(x))
df_train_norm['td_diff'] = (df_train_norm['td_diff'] - td_diff_mean) / td_diff_std
df_train_norm['pd_diff'] = (df_train_norm['pd_diff'] - pd_diff_mean) / pd_diff_std
print('normalized trend v price derviative diff')
print(np.corrcoef(df_train_norm['td_diff'], df_train_norm['pd_diff']))



#derivative setup
td_diff_mean = np.mean(df_train_norm['td_diff'])#.mean()
pd_diff_mean = np.mean(df_train_norm['pd_diff'])#.mean()

td_diff_std = np.std(df_train_norm['td_diff'])
pd_diff_std = np.std(df_train_norm['pd_diff'])

# df_train_norm.iloc[:,4:6] = df_train_norm.apply(lambda x: (x-np.mean(x))/np.std(x))
df_train_norm['td_diff'] = (df_train_norm['td_diff'] - td_diff_mean) / td_diff_std
df_train_norm['pd_diff'] = (df_train_norm['pd_diff'] - pd_diff_mean) / pd_diff_std
print('normalized trend v price derviative diff')
print(np.corrcoef(df_train_norm['td_diff'], df_train_norm['pd_diff']))



# ccf = sm.tsa.stattools.ccf(df_amd['td_diff'], df_amd['pd_diff'] )
# print(ccf[0:5])

#Extract interval ranges from training data
'''
pd_q75, pd_q60, pd_q40, pd_q25 = np.percentile(df_train_norm['price_diff'], [75, 60, 40, 25])
td_q75, td_q60, td_q40, td_q25 = np.percentile(df_train_norm['trend_diff'], [75, 60, 40, 25])
'''

#derivative
pd_q75, pd_q60, pd_q40, pd_q25 = np.percentile(df_train_norm['pd_diff'], [99.9, 60, 40, .1])
td_q75, td_q60, td_q40, td_q25 = np.percentile(df_train_norm['td_diff'], [99.9, 60, 40, .1])

def crossed_line(df):
    cross_lst = [False]
    for i in range(len(df['price_diff']) - 1):
        if(((df['price_diff'][i] < 0) & (df['price_diff'][i+1] > 0)) |\
        ((df['price_diff'][i] > 0) & (df['price_diff'][i+1] < 0))):
            cross_lst.append(True)
        else:
            cross_lst.append(False)
    return cross_lst
MY_RISK = 10
BASE_RISK = 10
MY_PORTFOLIO = 1000

#if price diff is greater than 75% interval and trend diff is less than 60%, short.
#if price diff is less than 25% interval and trend diff greater than 40%, buy.
#if trend diff is greater than 75% interval and price diff is less than 60%, buy
#if trend diff is less than 25% interval and price diff is greater than 40%, short
#if we have positions and the price crosses the rolling price (goes from pos to neg or neg to pos price diff),
#close positions
#else: hold
num_positions = 0
num_short = 0
num_long = 0
#price_diff and trend_diff aren't correlated, keep that in mind
#could change this logic to bet on volatility
'''
def simple_reco(df):#change these to 40 and 60 in next test
    if((df['price_diff'] > pd_q75) & (df['trend_diff'] < td_q60) & (df['trend_diff'] > td_q25)):
        return 'sell'
    elif((df['price_diff'] < pd_q25) & (df['trend_diff'] > td_q40) & (df['trend_diff'] < td_q75)):
        return 'buy'
    elif((df['trend_diff'] > td_q75) & (df['price_diff'] < pd_q60) & (df['price_diff'] > td_q25)):#can search for wild positive trend change, but this doesn't necessarily mean buy, can also be negative news
        return 'buy'
    elif((df['trend_diff'] < td_q25) & (df['price_diff'] > pd_q40) & (df['price_diff'] < td_q75)):
        return 'sell'
    # elif((df['crossed'] == True)):
    #     return 'neutralize'
    else:
        return 'neutralize'
'''
#deriv
def simple_reco(df):#change these to 40 and 60 in next test
    if((df['pd_diff'] > pd_q75) & (df['td_diff'] < td_q60) & (df['td_diff'] > td_q25)):
        return 'sell'
    elif((df['pd_diff'] < pd_q25) & (df['td_diff'] > td_q40) & (df['td_diff'] < td_q75)):
        return 'buy'
    elif((df['td_diff'] > td_q75) & (df['pd_diff'] < pd_q60) & (df['pd_diff'] > td_q25)):#can search for wild positive trend change, but this doesn't necessarily mean buy, can also be negative news
        return 'buy'
    elif((df['td_diff'] < td_q25) & (df['pd_diff'] > pd_q40) & (df['pd_diff'] < td_q75)):
        return 'sell'
    # elif((df['crossed'] == True)):
    #     return 'neutralize'
    else:
        return 'neutralize'


def reco(df, risk = MY_RISK):
    if df['td_diff'] > 0:
        return 'buy'
    elif df['td_diff'] < 0:
        return 'sell'
    else:
        return 'hold'


### Normalize price and trend differences in test data
#get z-score
df_test_norm = df_test.copy()

df_test_norm['td_diff'] = (df_test_norm['td_diff'] - td_diff_mean) / td_diff_std
df_test_norm['pd_diff'] = (df_test_norm['pd_diff'] - pd_diff_mean) / pd_diff_std

#populating values for train
df_norm = df_train_norm.append(df_test_norm)
#df_test = df_amd.iloc[-1:]

#crossed

df_norm['crossed'] = crossed_line(df_norm)

df_test_norm = df_norm.iloc[-1:]
df_train_norm = df_norm.iloc[:-1]


# df_test_norm['crossed'] = crossed_line(df_test_norm)

df_test_norm['recommendation'] = df_test_norm.apply(reco,risk = MY_RISK, axis = 1)
df_test_norm['simple_reco'] = df_test_norm.apply(simple_reco, axis = 1)
# print(df_test_norm)
df_test_norm.index = pd.to_datetime(df_test_norm.index)
tup_1 = (df_test_norm.iloc[0].name.strftime('%Y-%m-%d'), df_test_norm['price'][0], df_test_norm['recommendation'][0], df_test_norm['simple_reco'][0])
#tup_2 = (df_test_norm.iloc[1].name.strftime('%Y-%m-%d'), df_test_norm['price'][1], df_test_norm['recommendation'][1], df_test_norm['simple_reco'][1])
#cur.execute('DELETE FROM '+stock' WHERE name = ?', (fighter,))
cur.execute('INSERT INTO '+stock+' VALUES (?, ?, ?, ?)', tup_1)
#cur.execute('INSERT INTO '+stock+' VALUES (?, ?, ?, ?)', tup_2)
con.commit()



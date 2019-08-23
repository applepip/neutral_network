import pandas as pd
import time

from plot_bitcoin_chart import *

# get bitcoin market info from web site
bitcoin_market_info = pd.read_html("https://coinmarketcap.com/currencies/bitcoin/historical-data/?start=20130101&end="+time.strftime("%Y%m%d"), flavor='html5lib')[0]
# convert the date string to the correct date format
bitcoin_market_info = bitcoin_market_info.assign(Date=pd.to_datetime(bitcoin_market_info['Date']))
# convert to int
bitcoin_market_info['Volume'] = (pd.to_numeric(bitcoin_market_info['Volume'], errors='coerce').fillna(0))
# this will remove those asterisks
bitcoin_market_info.columns = bitcoin_market_info.columns.str.replace("*", "")
bitcoin_market_info.columns = [bitcoin_market_info.columns[0]] + ['bt_' + i for i in bitcoin_market_info.columns[1:]]
# look at the first few rows

print(bitcoin_market_info.head())
plot_bitcoin_chart(bitcoin_market_info)

bitcoin_market_info.to_csv('data/bitcoin.csv')


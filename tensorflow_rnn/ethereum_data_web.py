import time

from plot_coin_chart import *

# get bitcoin market info from web site
eth_market_info = pd.read_html("https://coinmarketcap.com/currencies/bitcoin/historical-data/?start=20130101&end="+time.strftime("%Y%m%d"), flavor='html5lib')[0]
# convert the date string to the correct date format
eth_market_info = eth_market_info.assign(Date=pd.to_datetime(eth_market_info['Date']))
# convert to int
eth_market_info['Volume'] = (pd.to_numeric(eth_market_info['Volume'], errors='coerce').fillna(0))
# this will remove those asterisks
eth_market_info.columns = eth_market_info.columns.str.replace("*", "")
eth_market_info.columns = [eth_market_info.columns[0]] + ['eth_' + i for i in eth_market_info.columns[1:]]
# look at the first few rows

print(eth_market_info.head())

plot_coin_chart(eth_market_info, 'eth')

eth_market_info.to_csv('data/ethereum.csv')


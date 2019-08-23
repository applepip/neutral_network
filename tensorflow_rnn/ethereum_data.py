import pandas as pd

from plot_bitcoin_chart import *

bitcoin_market_info = pd.read_csv('data/bitcoin.csv')

print(bitcoin_market_info.head())
plot_bitcoin_chart(bitcoin_market_info)


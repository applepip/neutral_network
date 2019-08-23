from plot_coin_chart import *

bitcoin_market_info = pd.read_csv('data/bitcoin.csv')

print(bitcoin_market_info.head())
plot_coin_chart(bitcoin_market_info, 'btc')

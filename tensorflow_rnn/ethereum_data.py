from plot_coin_chart import *

eth_market_info = pd.read_csv('data/ethereum.csv')

print(eth_market_info.head())
plot_coin_chart(eth_market_info, 'eth')


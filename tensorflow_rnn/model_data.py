from bitcoin_data import *
from ethereum_data import *
from plot_split_data import *

def merge_data(a, b, from_date):
    '''
    # merge the data with date
    :param a:
    :param b:
    :param from_date:
    :return:
    '''
    merged_data = pd.merge(a, b, on=['Date'])
    merged_data = merged_data[merged_data['Date'] >= from_date]

    for coins in ['btc_', 'eth_']:
        kwargs = {coins + 'day_diff': lambda x: (x[coins + 'Close'] - x[coins + 'Open']) / x[coins + 'Open']}
        merged_data = merged_data.assign(**kwargs)  #DataFrame.assign(**kwargs)将在DataFrame上计算并分配给新列。

    return merged_data

def split_training_data(market_info, split_date = '2018-01-01'):
    '''
    :param data: Pandas Dataframe
    :param training_size: proportion of the data to be used for training
    :return:
    '''

    coin_train_data = market_info[market_info['Date'] < split_date]
    coin_test_data = market_info[market_info['Date'] >= split_date]

    return coin_train_data, coin_test_data


market_info = merge_data(bitcoin_market_info, eth_market_info, '2016-01-01')

coin_train_data, coin_test_data = split_training_data(market_info)

print(coin_train_data.head())

# plot_split_data(market_info)




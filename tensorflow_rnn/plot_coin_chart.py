import matplotlib.pyplot as plt
import datetime
import pandas as pd
from pandas.plotting import register_matplotlib_converters

def plot_bitcoin_chart(bitcoin_market_info):
    '''
    根据数据绘制bitcoin的变化曲线
    :return:
    '''

    register_matplotlib_converters()

    volume = [0, 3000, 6000, 9000, 12000]

    # gridspec_kw={'height_ratios': [3, 1]} 设置子图在列上的分布比例是3比1
    fig, ax = plt.subplots(2, 1,  gridspec_kw={'height_ratios': [3, 1]})

    ax[0].set_ylabel('Closing Price ($)', fontsize=12)
    ax[0].set_xticks([datetime.date(i, j, 1) for i in range(2013, 2020) for j in [1, 7]])
    ax[0].set_xticklabels('')

    ax[1].set_ylabel('Volume ($ bn)', fontsize=12)
    ax[1].set_xticks([datetime.date(i, j, 1) for i in range(2013, 2020) for j in [1, 7]])
    ax[1].set_xticklabels([datetime.date(i, j, 1).strftime('%b %Y') for i in range(2013, 2020) for j in [1, 7]])

    ax[0].plot(pd.to_datetime(bitcoin_market_info['Date']), bitcoin_market_info['bt_Open'])
    ax[1].bar(pd.to_datetime(bitcoin_market_info['Date']).values, bitcoin_market_info['bt_Volume'].values)
    fig.tight_layout()
    plt.show()
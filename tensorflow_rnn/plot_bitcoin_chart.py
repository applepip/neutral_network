import matplotlib.pyplot as plt
import datetime

def plot_bitcoin_chart(bitcoin_market_info):
    '''
    根据数据绘制bitcoin的变化曲线
    :return:
    '''
    # gridspec_kw={'height_ratios': [3, 1]} 设置子图在列上的分布比例是3比1
    fig, ax = plt.subplots(2, 1,  gridspec_kw={'height_ratios': [3, 1]} )

    ax[0].set_ylabel('Closing Price ($)', fontsize=12)
    ax[0].set_xticks([datetime.date(i, j, 1) for i in range(2016, 2019) for j in [1, 5, 9]])
    ax[0].set_xticklabels('')

    ax[1].set_ylabel('Volume ($ bn)', fontsize=12)
    ax[1].set_yticks([int('%d000000000' % i) for i in range(10)])
    ax[1].set_yticklabels(range(10))
    ax[1].set_xticks([datetime.date(i, j, 1) for i in range(2016, 2019) for j in [1, 5, 9]])
    ax[1].set_xticklabels([datetime.date(i, j, 1).strftime('%b %Y') for i in range(2016, 2019) for j in [1, 5, 9]])

    ax[0].plot(bitcoin_market_info['Date'].dt.to_pydatetime(), bitcoin_market_info['bt_Open'])
    ax[1].bar(bitcoin_market_info['Date'].dt.to_pydatetime(), bitcoin_market_info['bt_Volume'])
    fig.tight_layout()
    plt.show()
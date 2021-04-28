from huobi.client.market import MarketClient
from huobi.constant import *
from huobi.utils import *
import pandas as pd

market_client = MarketClient(init_log=True)
interval = CandlestickInterval.MIN1
symbol = "ethusdt"
list_obj = market_client.get_candlestick(symbol, interval, 2000)
# print(list_obj)
LogInfo.output("---- {interval} candlestick for {symbol} ----".format(interval=interval, symbol=symbol))
LogInfo.output_list(list_obj)
Id=[]
High=[]
Low=[]
Open=[]
Close=[]
Count=[]
Amount=[]
Volume=[]
if list_obj and len(list_obj):
    for obj in list_obj:
        Id.append(obj.id)
        High.append(obj.high)
        Low.append(obj.low)
        Open.append(obj.open)
        Close.append(obj.close)
        Count.append(obj.count)
        Amount.append(obj.amount)
        Volume.append(obj.vol)
       # print(obj.id)
# 字典中的key值即为csv中列名
dataframe = pd.DataFrame({'Id': Id, 'High': High,'Low': Low,'Open': Open,'Close': Close,'Count': Count,'Amount': Amount,'Volume': Volume})

# 将DataFrame存储为csv,index表示是否显示行名，default=True
dataframe.to_csv(interval+symbol+".csv", index=False, sep=',')



















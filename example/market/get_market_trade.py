from huobi.client.market import MarketClient
from huobi.utils import *

market_client = MarketClient()
list_obj = market_client.get_market_trade(symbol="eosusdt")
LogInfo.output_list(list_obj)
print("---------------------")
print(list_obj[0].price)













# Create generic client instance and get the timestamp
from huobi.client.generic import GenericClient, CandlestickInterval
from huobi.client.market import MarketClient, LogInfo

generic_client = GenericClient()
timestamp = generic_client.get_exchange_timestamp()
print(timestamp)

# Create the market client instance and get the latest btcusdtĄŪs candlestick
# market_client = MarketClient()
# list_obj = market_client.get_candlestick("btcusdt", CandlestickInterval.MIN5, 10)
# LogInfo.output_list(list_obj)
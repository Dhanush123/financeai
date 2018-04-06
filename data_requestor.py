import requests
import json
import pandas as pd


def request_stock_data(stock_ids, start_date='20150101', end_date='20160101'):
	stocks_request= requests.get("https://www.blackrock.com/tools/hackathon/performance", 
	params= {'identifiers': ','.join(stock_ids),
			 'startDate': start_date,
			 'endDate': end_date,
			 'returnsType' :  "DAILY"})
	stocks_json = stocks_request.json()
	stocks_data = stocks_json['resultMap']['RETURNS']
	stock_to_date_to_price_dict = {stock_data['ticker']: extract_date_return(stock_data) for stock_data in stocks_data}
	df = pd.DataFrame(stock_to_date_to_price_dict)
	percentage_change = df.pct_change().iloc[1:, :]
	no_missing_percentage_change = percentage_change.fillna(0)
	no_missing_percentage_change[no_missing_percentage_change == 0] = 0.0001
	return no_missing_percentage_change

def extract_date_return(stock_data):
	return {date: stock_data_for_day['level'] for date, stock_data_for_day in stock_data['returnsMap'].items()}


tech_stocks = ['AAPL', 'MSFT', 'GOOGL']
airline_stocks = ['AAL', 'DAL', 'LUV']
misc_stocks = ['WFC', 'CVS', 'JNJ']




stocks = tech_stocks + airline_stocks + misc_stocks

new_stocks = ['MSFT',
'TSLA',
'GOOGL',
'CRM',
'TWTR',
'NFLX',
'FB',
'NVDA',
'AMD',
'TMUS'
]
# # airline_stocks + misc_stocks
df = request_stock_data(new_stocks, start_date='20150101', end_date='20170101')
df.to_csv('data/stocks2015-now.csv')

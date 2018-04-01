import requests
import json
import pandas as pd


def request_stock_data(stock_ids, start_date='20150101', end_date='20160101'):
	stocks_request= requests.get("https://www.blackrock.com/tools/hackathon/performance", 
	params= {'identifiers': ','.join(stock_ids),
			 'startDate': '20150101',
			 'endDate': '20160101',
			 'returnsType' :  "DAILY"})
	stocks_json = stocks_request.json()
	stocks_data = stocks_json['resultMap']['RETURNS']
	stock_to_date_to_price_dict = {stock_data['ticker']: extract_date_return(stock_data) for stock_data in stocks_data}
	df = pd.DataFrame(stock_to_date_to_price_dict)
	return df.pct_change().iloc[1:, :]

def compute_daily_percentage_change(col):
	offset_cols = col[:-1]
	return (col[1:] - offset_cols)/offset_cols

def extract_date_return(stock_data):
	return {date: stock_data_for_day['level'] for date, stock_data_for_day in stock_data['returnsMap'].items()}



print(request_stock_data(['AAPL', 'MSFT']))

import requests
import json

def request_stock_data(stock_ids, start_date='20150101', end_date='20160101'):
	stocks_request= requests.get("https://www.blackrock.com/tools/hackathon/performance", 
	params= {'identifiers': ','.join(stock_ids),
			 'startDate': '20150101',
			 'endDate': '20160101',
			 'returnsType' :  "DAILY"})
	stocks_json = stocks_request.json()
	print(stocks_json)
	stocks_data = stocks_json['resultMap']['RETURNS']
	return [extract_date_return(stock_data) for stock_data in stocks_data]

def extract_date_return(stock_data):
	return {date: stock_data_for_day['level'] for date, stock_data_for_day in stock_data['returnsMap'].items()}



print(request_stock_data(['AAPL', 'MSFT']))

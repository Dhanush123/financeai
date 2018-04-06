import numpy as np
import pandas as pd

tech_df = pd.read_csv('stocks.csv', index_col=0)

def reader_creator(data):
  def reader():
    for _, row in data.iterrows():
    	print(row)
    	yield np.array(row.as_matrix())
  return reader

for i in reader_creator(tech_df)():
	i
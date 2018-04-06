import plotly
import pandas as pd

plotly.tools.set_credentials_file(username='johnwu93', api_key='dQBFPRtzRIh5E1YAsekG')

from plotly.graph_objs import Scatter, Layout, Data, Bar
import plotly.plotly as py

x_top = [
    'AMD',
    'MSFT',
    'NVDA',
]
y_top = [
    0.004502401772289806,
    0.0030171221847832964,
    0.003013860017297881
]
x_bottom = [
    'FB',
    'TSLA',
    'NFLX',
    'TWTR',
    'CRM',
    'TMUS'
    ]

y_bottom = [
    0.002887567882715531
    , 0.002456205911982723
    , 0.0023413589535932477
    , 0.0005407114064733147
    , 0.0005169770988661801
    , 1.2541061923762813e-05
    ]

predicted_returns = pd.read_csv('average_returns.csv', index_col=0, header=None)
loser = Bar(x=['GOOGL'], y=[-0.00034161776722831235], marker=dict(color='red'), name='bad')
ok = Bar(
    x=x_bottom,
    y=y_bottom, marker=dict(
        color='blue'
    ), name='average')
exceptional = Bar(x=x_top, y=y_top, marker=dict(color='green'), name='exceptional')
data = [
    exceptional, ok,
    loser
]

print(py.plot(data, filename='predicted-expected-returns'))

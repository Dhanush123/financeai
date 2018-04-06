import paddle.v2 as paddle
import paddle.v2.dataset.uci_housing as uci_housing
import pandas as pd
import numpy as np

paddle.init(use_gpu=False, trainer_count=1)

tech_df = pd.read_csv('./data/stocks2015-now.csv', index_col=0)

num_stocks = len(tech_df.columns)


def simple_linear_regressor(num_dim, stock_id):
    x = paddle.layer.data(name='x{}'.format(stock_id), type=paddle.data_type.dense_vector(num_dim))
    y_predict = paddle.layer.fc(input=x,
                                size=1,
                                act=paddle.activation.Linear())
    y = paddle.layer.data(name='y{}'.format(stock_id), type=paddle.data_type.dense_vector(1))
    cost = paddle.layer.square_error_cost(input=y_predict, label=y)
    return (cost, y_predict)


def reader_creator(data, stock_id):
    def reader():
        for date_index in range(1, len(data)):
            predicted_stock = np.array([data.iloc[date_index, :][stock_id]])
            yield data.iloc[date_index - 1, :].as_matrix(), predicted_stock

    return reader


def train_regressor(df, stock_id):
    (cost, y_predict) = simple_linear_regressor(num_stocks, stock_id)
    parameters = paddle.parameters.create(cost)
    optimizer = paddle.optimizer.Momentum(momentum=0)
    trainer = paddle.trainer.SGD(cost=cost,
                                 parameters=parameters,
                                 update_equation=optimizer)

    trainer.train(
        reader=paddle.batch(
            paddle.reader.shuffle(
                reader_creator(df, stock_id), buf_size=500),
            batch_size=2),
        num_passes=30)
    return y_predict, parameters


def create_list_array(df):
    return [(np.array(row),) for _, row in df.iterrows()]


# y_predict, computed_params = train_regressor(tech_df, stock_id)
# predicted_input = create_list_array(tech_df)

# probs = paddle.infer(
#     output_layer=y_predict, parameters=computed_params, input=predicted_input)

# for i in xrange(len(probs)):
#     print "label=" + str(test_label[i][0]) + ", predict=" + str(probs[i][0])

class DlModel(object):
    def __init__(self, stock_id):
        self.stock_id = stock_id

    def fit(self, data):
        self.y_predict, self.parameters = train_regressor(data, self.stock_id)

    def predict(self, df):
        badly_formatted = paddle.infer(output_layer=self.y_predict, parameters=self.parameters,
                                       input=create_list_array(df))
        return [val[0] for val in badly_formatted]


def predict_daily_return(df, stock_id):
    model = DlModel(stock_id)
    n_day = 365
    first_year_df = df.iloc[:n_day, :]
    later_years_df = df.iloc[n_day:, :]
    model.fit(first_year_df)
    values = model.predict(later_years_df)
    return pd.Series(data=values, index=later_years_df.index)


def compute_returns(df):
    average_returns = dict()
    predicted_returns = dict()
    for stock_name in df:
        predicted_daily_returns = predict_daily_return(df, stock_name)
        predicted_returns[stock_name] = predicted_daily_returns
        expected_daily_return = np.mean(predicted_daily_returns)

        average_returns[stock_name] = expected_daily_return

    average_returns_stocks = pd.Series(average_returns).sort_values(ascending=False)
    predicted_daily_returns_stocks = pd.DataFrame(predicted_returns)
    return average_returns_stocks, predicted_daily_returns_stocks


average_returns, predicted_returns = compute_returns(tech_df)
average_returns.to_csv("average_returns.csv")
predicted_returns.to_csv("predicted_returns.csv")

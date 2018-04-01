import paddle.v2 as paddle
import paddle.v2.dataset.uci_housing as uci_housing
import pandas as pd
import numpy as np

paddle.init(use_gpu=False, trainer_count=1)


def simple_linear_regressor(num_dim):
	x = paddle.layer.data(name='x', type=paddle.data_type.dense_vector(2))
	y_predict = paddle.layer.fc(input=x,
	                                size=1,
	                                act=paddle.activation.Linear())
	y = paddle.layer.data(name='y', type=paddle.data_type.dense_vector(1))
	cost = paddle.layer.square_error_cost(input=y_predict, label=y)
	return (cost, y_predict)


tech_df = pd.read_csv('stocks.csv', index_col=0)


(cost, y_predict) = simple_linear_regressor(2)
parameters = paddle.parameters.create(cost)
optimizer = paddle.optimizer.Momentum(momentum=0)
trainer = paddle.trainer.SGD(cost=cost,
                             parameters=parameters,
                             update_equation=optimizer)

def reader_creator(data):
	def reader():
		for _, row in data.iterrows():
			yield row[:2].as_matrix(), np.array([row[2]])
	return reader


trainer.train(
    reader=paddle.batch(
        paddle.reader.shuffle(
            reader_creator(tech_df), buf_size=500),
        batch_size=2),
    num_passes=30)

test_label = []

def to_test_output(df):
	test_data = []
	for item in reader_creator(df)():
	    test_data.append(item[0])
	return test_data
	   

probs = paddle.infer(
    output_layer=y_predict, parameters=parameters, input=test_data)

for i in xrange(len(probs)):
    print "label=" + str(test_label[i][0]) + ", predict=" + str(probs[i][0])
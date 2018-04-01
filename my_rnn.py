import paddle.v2 as paddle
import paddle.v2.dataset.uci_housing as uci_housing
import pandas as pd
import numpy as np

def reader_creator(data):
  def reader():
    num_rows = len(data)
    for idx in range(1, num_rows):

      yield (data.iloc[idx - 1, :].as_matrix(), data.iloc[idx, :].as_matrix()) 
  return reader

def simple_rnn(input,
               size=None,
               name=None,
               reverse=False,
               rnn_bias_attr=None,
               act=None,
               rnn_layer_attr=None):
    def rnn_step(ipt):
       out_mem = paddle.layer.memory(name=name, size=size)
       rnn_out = paddle.layer.mixed(input = [paddle.layer.full_matrix_projection(input=ipt),
                                             paddle.layer.full_matrix_projection(input=out_mem)],
                                    name = name,
                                    bias_attr = rnn_bias_attr,
                                    act = act,
                                    layer_attr = rnn_layer_attr,
                                    size = size)
       return rnn_out
    return paddle.layer.recurrent_group(name='%s_recurrent_group' % name,
                                        step=rnn_step,
                                        reverse=reverse,
                                        input=input)

paddle.init(use_gpu=False, trainer_count=1)

x = paddle.layer.data(name='x', type=paddle.data_type.dense_vector(9))
y_predict = simple_rnn(x, 9, 'stock predictor')
next_time = paddle.layer.data(name='next_time', type=paddle.data_type.dense_vector(9)) # probably will fail

cost = paddle.layer.square_error_cost(input=y_predict, label=next_time)


parameters = paddle.parameters.create(cost)
optimizer = paddle.optimizer.Momentum(momentum=0)
trainer = paddle.trainer.SGD(cost=cost,
                             parameters=parameters,
                             update_equation=optimizer)

tech_df = pd.read_csv('stocks.csv', index_col=0)

def to_output(data):
  return [np.array(row) for _, row in data.iterrows()]


trainer.train(
    reader=paddle.batch(
        paddle.reader.shuffle(
            reader_creator(tech_df), buf_size=500),
        batch_size=2),
    num_passes=30)

probs = paddle.infer(
    output_layer=y_predict, parameters=parameters, input=to_output(tech_df))
import paddle.v2 as paddle 

# Initialize PaddlePaddle.
paddle.init(use_gpu=False, trainer_count=1)

# def simple_rnn(input,
#                size=None,
#                name=None,
#                reverse=False,
#                rnn_bias_attr=None,
#                act=None,
#                rnn_layer_attr=None):
#     def __rnn_step__(ipt):
#        out_mem = paddle.layer.memory(name=name, size=size)
#        rnn_out = paddle.layer.mixed(input = [paddle.layer.full_matrix_projection(input=ipt),
#                                              paddle.layer.full_matrix_projection(input=out_mem)],
#                                     name = name,
#                                     bias_attr = rnn_bias_attr,
#                                     act = act,
#                                     layer_attr = rnn_layer_attr,
#                                     size = size)
#        return rnn_out
#     return paddle.layer.recurrent_group(name='%s_recurrent_group' % name,
#                                         step=__rnn_step__,
#                                         reverse=reverse,
#                                         input=input)

# x = paddle.layer.data(name='x', type=paddle.data_type.dense_vector(12))
paddle.layer.mixed
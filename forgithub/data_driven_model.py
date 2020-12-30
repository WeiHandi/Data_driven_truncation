import numpy as np
import tensorflow as tf
import tensorlayer as tl
import scipy.io as sio
from tensorboard.backend.event_processing import event_accumulator
from operator_feat_builder import feat_builder
from operator_standard import standard
from operator_standard import power_standard
input_indice = ['total_l1', 'total_l2', 'total_l3', 'total_e1', 'total_e2', 'total_e3', 'total_w1',
                'total_w2', 'total_w3', 'x', 'y', 'r_t', 'theta_t', 'r_a', 'theta_a', 'r_tv', 'r_th']
output_indice = ['total_l2', 'total_w2', 'total_e2/l2', 'x']
num_train_rate = 0.8
num_cv_rate = 0.1
num_test_rate = 0.1
p = 1

file_name = ['425', '450', '475', '500', '525', '550', '575', '600', '625']
property_num = ['l2', 'w2', 'e2']
depth = np.array([425, 450, 475, 500, 525, 550, 575, 600, 625])


def normalization(input_vec, output_vec):
    feat_vec = feat_builder(input_vec)
    [feat_vec_st, array_max, array_min] = standard(feat_vec)
    # input_vec_st, _, _ = standard(input_vec)
    output_vec_st, a_max, a_min = power_standard(output_vec, p)
    array_max = array_max.reshape([array_max.shape[0], 1])
    array_min = array_min.reshape([array_min.shape[0], 1])
    a_max = a_max.reshape([a_max.shape[0], 1])
    a_min = a_min.reshape([a_min.shape[0], 1])
    return feat_vec_st, output_vec_st, array_max, array_min, a_max, a_min


def init_Data(feat_vec_st, output_vec_st, num_samp, num_train_rate, num_cv_rate, num_test_rate):
    num_train = int(num_train_rate*num_samp)
    num_cv = int(num_cv_rate*num_samp)+num_train
    num_test = int(num_test_rate*num_samp)+num_cv
    x_train = feat_vec_st[0:num_train-1, :]
    y_train = output_vec_st[0:num_train-1, :]
    x_val = feat_vec_st[num_train:num_cv-1, :]
    y_val = output_vec_st[num_train:num_cv-1, :]
    x_test = feat_vec_st[num_cv:num_test-1, :]
    y_test = output_vec_st[num_cv:num_test-1, :]
    return x_train, y_train, x_val, y_val, x_test, y_test


def init_PlaceHolder(x_train, y_train):
    with tf.name_scope("inputs"):
        x = tf.placeholder(tf.float32, shape=[
            None, x_train.shape[1]], name='x')
        y_ = tf.placeholder(tf.float32, shape=[None, 1], name='y_')
    return x, y_


class NetworkStructure:
    def __init__(self, network, trainMethod, loss, acc):
        self.network = network
        self.trainMethod = trainMethod
        self.loss = loss
        self.acc = acc


def buildNerwork(x_placeholder, y_placeholder):
    tl.layers.set_name_reuse(True)
    network = tl.layers.InputLayer(x_placeholder, name='input_layer')
    network = tl.layers.DropoutLayer(network, keep=1, name='drop1')
    network = tl.layers.DenseLayer(network, n_units=128,
                                   act=tf.nn.relu, name='relu1')
    network = tl.layers.DropoutLayer(network, keep=1, name='drop2')
    network = tl.layers.DenseLayer(
        network, n_units=512, act=tf.nn.relu, name='relu2')
    network = tl.layers.DenseLayer(
        network, n_units=1, act=tf.identity, name='output_layer')
    # define the cost function
    y = network.outputs
    cost = tf.nn.l2_loss(y-y_placeholder, name='cost')
    train_params = network.all_params
    trainMethod = tf.train.AdamOptimizer(learning_rate=0.0001, beta1=0.9, beta2=0.999,
                                         epsilon=1e-06, use_locking=False).minimize(cost, var_list=train_params)
    acc = None
    MyNetwork = NetworkStructure(network, trainMethod, cost, acc)
    return MyNetwork








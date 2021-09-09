#!/usr/local/bin/python
from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf

from utils import *
from models import MCGCN
import save_dict
import numpy as np
import matplotlib.pyplot as plt
from pykalman import KalmanFilter
# Set random seed

seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)
airport_num = 60

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('model', 'mcgcn', 'Model string.')
flags.DEFINE_float('learning_rate', 0.00003, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 50, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 16, 'Number of units in hidden layer 1.')
flags.DEFINE_float('dropout', 0.3, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 1e-3, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_float('alpha', 0.1, 'alpha balancing MCGCN')


checkpt_file = 'result/mcgcn_'+'.ckpt'
# Load data


data_adj = save_dict.load_obj('adj_list_1hour_0401_new')
data_features = save_dict.load_obj('input_feature_1hour_0401_transpose_new')
data_labels = save_dict.load_obj('delay_labels_1hour_0401_new')
data_pattern_dis = save_dict.load_obj('pattern_dis_all_0401_new')
data_geo_neighbor_adj = save_dict.load_obj('geo_neighbor_adj')

data_adj = np.array(data_adj)
data_features = np.array(data_features)
data_labels = np.array(data_labels)
data_pattern_dis = np.array(data_pattern_dis)


data_labels = np.transpose(data_labels)





def Kalman1D(observations, damping=1):
    # To return the smoothed time series data
    observation_covariance = damping
    initial_value_guess = observations[0]
    transition_matrix = 1
    transition_covariance = 0.1

    kf = KalmanFilter(
            initial_state_mean=initial_value_guess,
            initial_state_covariance=observation_covariance,
            observation_covariance=observation_covariance,
            transition_covariance=transition_covariance,
            transition_matrices=transition_matrix
        )
    pred_state, state_cov = kf.smooth(observations)
    return pred_state

ccc = []
for item in data_labels:
    aa = Kalman1D(item)
    ccc.append(aa)
data_labels = np.array(ccc)


adj, features_all, pattern_dis, geo_neighbor_adj, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(data_adj, data_features, data_pattern_dis, data_geo_neighbor_adj, data_labels)
A_geo = preprocess_geo(geo_neighbor_adj)

data_labels_train = data_labels[:1794]
data_labels_val = data_labels[1794:1794 + 384]
data_labels_test = data_labels[-384:]

data_train_stats = {'mean': np.mean(data_labels_train), 'std': np.std(data_labels_train)}
data_val_stats = {'mean': np.mean(data_labels_val), 'std': np.std(data_labels_val)}
data_test_stats = {'mean': np.mean(data_labels_test), 'std': np.std(data_labels_test)}


if FLAGS.model == 'mcgcn':
    support = [A_geo, A_geo]
    num_supports = 2
    model_func = MCGCN


r_time1 = time.time()
# Define placeholders
placeholders = {
    'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
    'features': tf.sparse_placeholder(tf.float32, shape=tf.constant((airport_num, 4), dtype=tf.int64)),
    'tuopu_adj': tf.placeholder(tf.float32, shape=(airport_num, airport_num)),
    'pattern_dis': tf.placeholder(tf.float32, shape=(airport_num, airport_num)),
    'labels': tf.placeholder(tf.float32, shape=(airport_num, 1)),
    'global_': tf.placeholder(tf.int32),
    'dropout': tf.placeholder_with_default(0., shape=()),
    'num_features_nonzero': tf.placeholder(tf.int32),

}

# Create model
model = model_func(placeholders, input_dim=4, logging=True)
sess = tf.Session()


# Define model evaluation function
def evaluate(features, support, pattern_dis, tuopu, labels, placeholders):
    t_test = time.time()
    feed_dict_val = construct_feed_dict(features, support, pattern_dis, tuopu, labels, placeholders)
    outs_val = sess.run([model.loss, model.accuracy, model.cross_entropy_loss], feed_dict=feed_dict_val)
    return outs_val[0], outs_val[1], outs_val[2], (time.time() - t_test)


def mae_loss_np(preds, labels):
    loss = np.abs(np.subtract(preds, labels))
    return np.mean(loss)

def mse_loss_np(preds, labels):
    loss = np.square(np.subtract(preds, labels))
    return np.mean(loss)

def rmse_loss_np(preds, labels):
    loss_mse = mse_loss_np(preds, labels)
    loss = np.sqrt(loss_mse)
    return loss

def mape_loss_np(preds, labels):
    mask = labels != 0
    return np.fabs((labels[mask] - preds[mask]) / labels[mask]).mean()


def smape_loss_np(preds, labels):
    return 2.0 * np.mean(np.abs(preds - labels) / (np.abs(preds) + np.abs(labels)))


def r2_loss_np(preds, labels):
    return 1 - mse_loss_np(preds, labels) / np.std(labels)


def z_score_label(x, mu, sigma):
    return (x - mu) / sigma


def evaluate_test(norm_features, support, pattern_dis, label_mu, label_sigma, origin_labels, tuopu, norm_labels, placeholders):
    t_test = time.time()
    feed_dict_val = construct_feed_dict(norm_features, support, pattern_dis, tuopu, norm_labels, placeholders)
    outs_labels = sess.run(model.outputs, feed_dict=feed_dict_val)
    outs_labels = z_inverse(outs_labels, label_mu, label_sigma)

    mse_out = mse_loss_np(outs_labels, origin_labels)
    mae_out = mae_loss_np(outs_labels, origin_labels)
    rmse_out = rmse_loss_np(outs_labels, origin_labels)
    mape_out = mape_loss_np(outs_labels, origin_labels)
    smape_out = smape_loss_np(outs_labels, origin_labels)

    return mse_out, mae_out, rmse_out, mape_out, smape_out, (time.time() - t_test)



# Init variables
sess.run(tf.global_variables_initializer())


vacc_mx = 0.0
vlss_mn = np.inf
curr_step = 0
saver = tf.train.Saver()



train_mask_list = list(train_mask)
val_mask_list = list(val_mask)
test_mask_list = list(test_mask)

train_total = train_mask_list.count(True)
val_total = val_mask_list.count(True)
test_total = test_mask_list.count(True)


# Train model
epoch_train_result = []
epoch_val_result = []

for epoch in range(FLAGS.epochs):#FLAGS.epochs
    step_train_loss = 0
    step_train_acc = 0
    step_train_cross_loss = 0
    step_val_loss = 0
    step_val_acc = 0
    step_val_rmse = 0
    step_val_cross_loss = 0

    t = time.time()
    for i in range(train_total):
        batch_features = features_all[i]
        batch_support = support
        batch_pattern_dis = pattern_dis[i]
        batch_tuopu = adj[i]
        batch_y_train = y_train[i]


        batch_features = z_score(batch_features)
        batch_y_train = z_score_label(batch_y_train, data_train_stats['mean'], data_train_stats['std'])

        batch_tuopu = preprocess_gcn(batch_tuopu)

        batch_features = preprocess_features(batch_features)

        batch_y_train = batch_y_train.reshape(airport_num, 1)


        # Construct feed dictionary
        feed_dict = construct_feed_dict(batch_features, batch_support, batch_pattern_dis, batch_tuopu, batch_y_train, placeholders)
        feed_dict.update({placeholders['dropout']: FLAGS.dropout})



        # Training
        outs = sess.run([model.opt_op, model.loss, model.accuracy, model.cross_entropy_loss, model.outputs], feed_dict=feed_dict)

        step_train_loss += outs[1]
        step_train_acc += outs[2]
        step_train_cross_loss += outs[3]

    print("Epoch:%d" % (epoch + 1), "train_mse_loss=", "{:.5f}".format(step_train_loss / train_total),
          "train_mae_loss=", "{:.5f}".format(step_train_acc / train_total), "train_rmse_loss=", "{:.5f}".format(step_train_cross_loss / train_total), "time=", "{:.5f}".format(time.time() - t))

    epoch_train_result.append(step_train_loss / train_total)



    # Validation
    for i in range(val_total):
        batch_features = features_all[train_total+i]
        batch_support = support
        batch_pattern_dis = pattern_dis[train_total+i]
        batch_tuopu = adj[train_total+i]
        batch_y_val = y_val[train_total+i]

        batch_features = z_score(batch_features)
        batch_y_val = z_score_label(batch_y_val, data_val_stats['mean'], data_val_stats['std'])

        batch_tuopu = preprocess_gcn(batch_tuopu)

        batch_features = preprocess_features(batch_features)

        batch_y_val = batch_y_val.reshape(airport_num, 1)

        cost, val_acc, val_loss, duration = evaluate(batch_features, batch_support, batch_pattern_dis, batch_tuopu, batch_y_val, placeholders)
        step_val_loss += cost
        step_val_acc += val_acc
        step_val_rmse += val_loss

    print("Epoch:%d" % (epoch + 1), "val_mse_loss=", "{:.5f}".format(step_val_loss / val_total),
          "val_mae_loss=", "{:.5f}".format(step_val_acc / val_total), "val_rmse_loss=", "{:.5f}".format(step_val_rmse / val_total), "time=", "{:.5f}".format(time.time() - t))

    epoch_val_result.append(step_val_loss / val_total)

    if epoch == FLAGS.epochs - 1:
        saver.save(sess, checkpt_file)


saver.restore(sess, checkpt_file)
# Testing
step_test_loss = 0
step_test_acc = 0
step_test_rmse = 0
step_test_mape = 0
step_test_smape = 0
t = time.time()

for i in range(test_total):
    batch_features = features_all[train_total + val_total + i]
    batch_support = support
    batch_pattern_dis = pattern_dis[train_total + val_total + i]
    batch_tuopu = adj[train_total + val_total + i]
    batch_y_test_origin = y_test[train_total + val_total + i]

    batch_features = z_score(batch_features)

    batch_y_test = z_score_label(batch_y_test_origin, data_test_stats['mean'], data_test_stats['std'])

    batch_features = preprocess_features(batch_features)

    batch_tuopu = preprocess_gcn(batch_tuopu)

    batch_y_test = batch_y_test.reshape(airport_num, 1)



    test_mse, test_mae, test_rmse, test_mape, test_smape, test_duration  = evaluate_test(batch_features, batch_support, batch_pattern_dis, data_test_stats['mean'], data_test_stats['std'],batch_y_test_origin, batch_tuopu, batch_y_test, placeholders)
    step_test_loss += test_mse
    step_test_acc += test_mae
    step_test_rmse += test_rmse
    step_test_mape += test_mape
    step_test_smape += test_smape



print("Test result:", "mse_loss=", "{:.5f}".format(step_test_loss / test_total),
          "mae_loss=", "{:.5f}".format(step_test_acc / test_total), "rmse_loss=", "{:.5f}".format(step_test_rmse / test_total), "mape_loss=", "{:.5f}".format(step_test_mape / test_total),  "smape_loss=", "{:.5f}".format(step_test_smape / test_total), "time=", "{:.5f}".format(time.time() - t))


sess.close()
r_time2 = time.time()

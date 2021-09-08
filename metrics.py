#!/usr/local/bin/python
import tensorflow as tf


def masked_softmax_cross_entropy(preds, labels, mask):
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=labels)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    loss *= mask
    return tf.reduce_mean(loss)


def masked_accuracy(preds, labels, mask):
    correct_prediction = tf.equal(tf.argmax(preds, 1), tf.argmax(labels, 1))
    accuracy_all = tf.cast(correct_prediction, tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    accuracy_all *= mask
    return tf.reduce_mean(accuracy_all)



def softmax_cross_entropy(preds, labels):
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=labels)
    return tf.reduce_mean(loss)


def accuracy(preds, labels):
    correct_prediction = tf.equal(preds, labels)
    accuracy_all = tf.cast(correct_prediction, tf.float32)
    return tf.reduce_mean(accuracy_all)



def mae_loss(preds, labels):
    loss = tf.abs(tf.subtract(preds, labels))
    return tf.reduce_mean(loss)

def mse_loss(preds, labels):
    loss = tf.square(tf.subtract(preds, labels))
    return tf.reduce_mean(loss)

def rmse_loss(preds, labels):
    loss_mse = mse_loss(preds, labels)
    loss = tf.sqrt(loss_mse)
    return loss


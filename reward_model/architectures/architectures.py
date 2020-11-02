from reward_model.layers.layers import *
import tensorflow as tf
import numpy as np

def deepcrawl_input():

    state = tf.compat.v1.placeholder(tf.float32, [None, 10, 10, 52], name='state')
    local_state = tf.compat.v1.placeholder(tf.float32, [None, 5, 5, 52], name='local_state')
    local_two_state = tf.compat.v1.placeholder(tf.float32, [None, 3, 3, 52], name='local_two_state')
    agent_stats = tf.compat.v1.placeholder(tf.int32, [None, 14], name='agent_stats')
    target_stats = tf.compat.v1.placeholder(tf.int32, [None, 4], name='target_stats')

    state_n = tf.compat.v1.placeholder(tf.float32, [None, 10, 10, 52], name='state_n')
    local_state_n = tf.compat.v1.placeholder(tf.float32, [None, 5, 5, 52], name='local_state_n')
    local_two_state_n = tf.compat.v1.placeholder(tf.float32, [None, 3, 3, 52], name='local_two_state_n')
    agent_stats_n = tf.compat.v1.placeholder(tf.int32, [None, 14], name='agent_stats_n')
    target_stats_n = tf.compat.v1.placeholder(tf.int32, [None, 4], name='target_stats_n')

    act = tf.compat.v1.placeholder(tf.int32, [None, 1], name='act')

    states = [state, local_state, local_two_state, agent_stats, target_stats]
    states_n = [state_n, local_state_n, local_two_state_n, agent_stats_n, target_stats_n]

    return states, act, states_n

def deepcrawl_obs_to_state(obs):

    global_batch = np.stack([np.asarray(state['global_in']) for state in obs])

    local_batch = np.stack([np.asarray(state['local_in']) for state in obs])

    local_two_batch = np.stack([np.asarray(state['local_in_two']) for state in obs])

    agent_stats_batch = np.stack([np.asarray(state['agent_stats'][:14]) for state in obs])

    target_stats_batch = np.stack([np.asarray(state['target_stats'][:4]) for state in obs])

    return global_batch, local_batch, local_two_batch, agent_stats_batch, target_stats_batch


def potions_net(states, act=None, with_action = False, actions_size = 10):

    state = states[0]
    conv_10 = conv_layer_2d(state, 32, [1, 1], name='conv_10', activation=tf.nn.tanh)
    conv_11 = conv_layer_2d(conv_10, 32, [3, 3], name='conv_11', activation=tf.nn.leaky_relu)
    conv_12 = conv_layer_2d(conv_11, 64, [3, 3], name='conv_12', activation=tf.nn.leaky_relu)

    all_flat = tf.reshape(conv_12, [-1, 10 * 10 * 64])

    all_flat = linear(all_flat, 100, name='fc1', activation=tf.nn.leaky_relu)

    if with_action:
        hot_acts = tf.one_hot(act, actions_size)
        hot_acts = tf.reshape(hot_acts, [-1, actions_size])
        all_flat = tf.concat([all_flat, hot_acts], axis=1)

    fc2 = linear(all_flat, 100, name='fc2', activation=tf.nn.leaky_relu)
    output = linear(fc2, 1, name='out')

    return output

def minigrid_input():

    state = tf.compat.v1.placeholder(tf.int32, [None, 15, 15], name='state')
    local_state = tf.compat.v1.placeholder(tf.int32, [None, 7, 7], name='local_state')

    state_n = tf.compat.v1.placeholder(tf.int32, [None, 15, 15], name='state_n')
    local_state_n = tf.compat.v1.placeholder(tf.int32, [None, 7, 7], name='local_state_n')

    act = tf.compat.v1.placeholder(tf.int32, [None, 1], name='act')

    states = [state, local_state]
    states_n = [state_n, local_state_n]

    return states, act, states_n


def minigrid_obs_to_state(obs):
    global_batch = np.stack([np.asarray(state['global_in']) for state in obs])

    local_batch = np.stack([np.asarray(state['local_in']) for state in obs])

    return global_batch, local_batch

def multiroom_net(states, act=None, with_action = False, actions_size = 10):

    state = states[0]
    conv_10 = embedding(state, 11, 32, name='embs_51')
    conv_11 = conv_layer_2d(conv_10, 32, [3, 3], name='conv_11', activation=tf.nn.leaky_relu)
    conv_11 = tf.nn.max_pool(conv_11, ksize=[2, 2], strides=2, padding='VALID')
    conv_11 = conv_layer_2d(conv_11, 32, [3, 3], name='conv_12', activation=tf.nn.leaky_relu)
    conv_11 = tf.nn.max_pool(conv_11, ksize=[2, 2], strides=2, padding='VALID')

    all_flat = tf.reshape(conv_11, [-1, 3 * 3 * 32])

    fc1 = linear(all_flat, 32, name='fc1', activation=tf.nn.leaky_relu)

    if with_action:
        hot_acts = tf.one_hot(act, actions_size)
        hot_acts = tf.reshape(hot_acts, [-1, actions_size])
        fc1 = tf.concat([fc1, hot_acts], axis=1)

    fc2 = linear(fc1, 32, name='fc2', activation=tf.nn.leaky_relu)

    output = linear(fc2, 1, name='out')

    return output


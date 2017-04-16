
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import time

def showAnimation(list_of_array_2d, fps):
    plt.ion()
    img = None
    for m in list_of_array_2d:
        t = time.time()
        plt.grid(False)
        if img is None:
            img = plt.imshow(m, interpolation='nearest')
        else:
            img.set_data(m)
        plt.draw()
        plt.pause(max(1.0 / fps - (time.time() - t), 0.1))

def initialBoard():
    board_state = np.zeros((10, 10), np.float32)
    board_state[2,1] = 1.0
    board_state[2,2] = 1.0
    board_state[2,3] = 1.0

    board_state[3,2] = 1.0
    board_state[3,3] = 1.0
    board_state[3,4] = 1.0

    return board_state

def createGraph(board_shape):
    in_state = tf.placeholder(tf.float32, shape=board_shape)

    new_values = []
    for x in range(1, board_shape[0] - 1):
        for y in range(1, board_shape[1] - 1):
            ss = tf.strided_slice(in_state, begin=[x-1, y-1], end=[x+2, y+2], strides=[1,1])
            sum = tf.reduce_sum(ss)

            def c1(): return tf.constant(1.0)
            def c0(): return tf.constant(0.0)

            def f1(): return tf.cond(tf.logical_and(tf.greater(sum, 2.9), tf.less(sum, 4.1)), c1, c0)
            def f0(): return tf.cond(tf.logical_and(tf.greater(sum, 2.9), tf.less(sum, 3.1)), c1, c0)

            r = tf.cond(tf.greater(in_state[x, y], 0.9), f1, f0)
            new_values.append(r)

    out_state = tf.reshape(tf.stack(new_values), ([board_shape[0]-2, board_shape[1]-2]))

    paddings = [[1, 1,], [1, 1]]
    out_state = tf.pad(out_state, paddings, "CONSTANT")

    return in_state, out_state


board = initialBoard()
seen_states = [board]

with tf.Graph().as_default():
    in_state, outState = createGraph(board.shape)

    with tf.Session() as sess:
        file_writer = tf.summary.FileWriter('train', sess.graph)
        for _ in range(0, 100):
            next_state = sess.run(outState, feed_dict={in_state: board})
            seen_states.append(next_state)
            board = next_state

showAnimation(seen_states, 2)

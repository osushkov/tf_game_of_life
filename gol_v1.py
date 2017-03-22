
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

def countNeighbours(board, tx, ty):
    num = 0
    for x in [max(tx - 1, 0), tx, min(tx + 1, board.shape[0] - 1)]:
        for y in [max(ty - 1, 0), ty, min(ty + 1, board.shape[1] - 1)]:
            num += board[x, y]
    return num - board[tx, ty]


def nextTimestemp(prevBoard):
    nextBoard = np.zeros_like(prevBoard)

    for x in range(1, nextBoard.shape[0] - 1):
        for y in range(1, nextBoard.shape[1] - 1):
            neighbours = countNeighbours(prevBoard, x, y)
            if prevBoard[x, y] == 1:
                nextBoard[x, y] = 1 if (neighbours >= 2 and neighbours <= 3) else 0
            else:
                nextBoard[x, y] = 1 if neighbours == 3 else 0

    return nextBoard


def createInitial(size):
    result = np.zeros([size, size])
    result[1:-1, 1:-1] = np.random.randint(2, size=(size - 2, size - 2))
    return result


BOARD_SIZE = 50
state = createInitial(BOARD_SIZE)


#
# plt.ion()
# for i in range(0, 1000):
#     plt.imshow(state, cmap='Greys', interpolation='nearest')
#     plt.draw()
#     _ = raw_input("Press [enter] to continue.")
#
#     state = nextTimestemp(state)


x = tf.constant([35, 40, 45], name='x')
y = tf.Variable(x + 5, name='y')


model = tf.global_variables_initializer()

with tf.Session() as session:
	session.run(model)
	print(session.run(y))


import numpy as np
import matplotlib.pyplot as pypl
import time
np.random.seed(int(time.time()))

u = np.arange(0,10,0.01)
k = np.tan(np.sin(u) + np.cos(u))

factor = 4 * np.max(k)
rand_values = np.random.rand(k.shape[0])
X = np.array(factor*(rand_values-np.mean(rand_values)))

labels = np.zeros(X.shape[0])
labels[X >= k] = 1

ix = np.array(range(X.shape[0]))
full_X = np.column_stack((ix, X))
data = np.column_stack((full_X, labels))

X = data[:,[0,1]]
y = data[:,2].astype('int32')

pos = np.array(y == 1)
neg = np.array(y == 0)

pypl.scatter(X[pos, 0], X[pos, 1], c='b', label='+ive examples')
pypl.scatter(X[neg, 0], X[neg, 1], c='r', label='-ive examples')
pypl.plot(k, c='g', label='true dec. bound.')
pypl.xlabel('Training examples feature# 1')
pypl.ylabel('Training examples feature# 2')
pypl.legend(loc='upper right')
pypl.show()

m, n = X.shape
ins = 2
hid1_units = 20
outs = 2
epsilon = 1e-3

Theta1 = epsilon * np.random.randn(ins,hid1_units)
b1 = np.zeros((1,hid1_units))

Theta2 = epsilon * np.random.randn(hid1_units,outs)
b2 = np.zeros((1,outs))

epochs = 100
batch_size = 100
alpha = 1e-3

for e in range(epochs):

    A2 = np.maximum(0, X.dot(Theta1) + b1)
    A3 = A2.dot(Theta2) + b2

    exp_A3 = np.exp(A3)
    # exp_A3 /= np.max(exp_A3, axis=0, keepdims=True)
    prob = exp_A3 / np.sum(exp_A3, axis=1, keepdims=True)
    loss = -1 / m * np.sum(np.log(prob[range(m), y]))
    print('epoch#', e, ' loss = ', loss)

    dL_dA3 = prob
    dL_dA3[range(m), y] -= 1
    dL_dA3 /= m

    dL_db2 = dL_dA3.sum(axis=0, keepdims=True)
    dL_dTh2 = np.dot(A2.T, dL_dA3)

    dL_dA2 = np.dot(dL_dA3, Theta2.T)
    dL_dA2[A2 <= 0] = 0

    dL_db1 = dL_dA2.sum(axis=0, keepdims=True)
    dL_dTh1 = np.dot(X.T, dL_dA2)

    Theta1 += -alpha * dL_dTh1
    b2 += -alpha * dL_db2
    Theta2 += -alpha * dL_dTh2
    b1 += -alpha * dL_db1

















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

pypl.figure(num=1)
pypl.scatter(X[pos, 0], X[pos, 1], c='b', label='+ive examples')
pypl.scatter(X[neg, 0], X[neg, 1], c='r', label='-ive examples')
pypl.plot(k, c='g', label='true dec. bound.')
pypl.xlabel('Training examples feature# 1')
pypl.ylabel('Training examples feature# 2')
pypl.legend(loc='upper right')
pypl.show()

m, n = X.shape
ins = 2
hid1_units = 40
hid2_units = 20
outs = 2
epsilon = 1e-3

Theta1 = epsilon * np.random.randn(ins,hid1_units)
b1 = np.zeros((1,hid1_units))

Theta2 = epsilon * np.random.randn(hid1_units,hid2_units)
b2 = np.zeros((1,hid2_units))

Theta3 = epsilon * np.random.randn(hid2_units,outs)
b3 = np.zeros((1,outs))

epochs = 40000
batch_size = 100
alpha = 1e-4
losses = []
e = 1
#for e in range(epochs):
while True:
    A2 = np.maximum(0, X.dot(Theta1) + b1)
    A3 = np.maximum(0, A2.dot(Theta2) + b2)
    A4 = A3.dot(Theta3) + b3

    exp_scores = np.exp(A4)
    prob = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    loss = -1 / m * np.sum(np.log(prob[range(m), y]))
    print('epoch#', e, ' loss = ', loss)
    e += 1
    losses.append(loss)

    dL_dA4 = prob
    dL_dA4[range(m), y] -= 1
    dL_dA4 /= m

    dL_db3 = dL_dA4.sum(axis=0, keepdims=True)
    dL_dTh3 = np.dot(A3.T, dL_dA4)

    dL_dA3 = np.dot(dL_dA4, Theta3.T)
    dL_dA3[A3 <= 0] = 0

    dL_db2 = dL_dA3.sum(axis=0, keepdims=True)
    dL_dTh2 = np.dot(A2.T, dL_dA3)

    dL_dA2 = np.dot(dL_dA3, Theta2.T)
    dL_dA2[A2 <= 0] = 0

    dL_db1 = dL_dA2.sum(axis=0, keepdims=True)
    dL_dTh1 = np.dot(X.T, dL_dA2)

    Theta1 += -alpha * dL_dTh1
    b1 += -alpha * dL_db1

    Theta2 += -alpha * dL_dTh2
    b2 += -alpha * dL_db2

    Theta3 += -alpha * dL_dTh3
    b3 += -alpha * dL_db3


train_acc = 100 * np.mean(np.argmax(A4, axis=1) == y)
print('Training accuracy = ', train_acc)

pypl.figure(2)
pypl.plot(losses, c='r')
pypl.xlabel('iteration')
pypl.ylabel('cost')
pypl.show()












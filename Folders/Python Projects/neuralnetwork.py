
import numpy as np
import matplotlib.pyplot as pypl
#import matplotlib.rcsetup as rcsetup
import time
np.random.seed(int(time.time()))

k = np.tanh(np.sin(np.arange(0,10,0.01))**2)
factor = 4 * np.max(k)
rand_values = np.random.rand(k.shape[0])
X = factor*(rand_values-np.mean(rand_values))

pos = np.array(np.where(X >= k))
neg = np.array(np.where(X < k))

pos_ex = np.array(X[pos])
neg_ex = np.array(X[neg])

y = np.zeros(X.shape)
y[pos] = 1
y[neg] = 0

labels   = np.concatenate((y[pos].T, y[neg].T), axis=0)
positive = np.concatenate((pos.T, pos_ex.T), axis=1)
negative = np.concatenate((neg.T, neg_ex.T), axis=1)
#print(indx.shape, exam.shape)
full_X = np.concatenate((positive, negative), axis=0)
data = np.concatenate((full_X, y), axis = 1)
#print(full_X)

subset = (data[:,2] == 1)

#pypl.scatter(pos, pos_ex, c='b', label='+ive examples')
#pypl.scatter(neg, neg_ex, c='r', label='-ive examples')
pypl.scatter(full_X[subset,0], full_X[subset,1], c='r', label='-ive examples')

pypl.plot(k, c='g', label='true dec. bound.')
pypl.xlabel('Training examples feature 1')
pypl.ylabel('Training examples feature 2')
pypl.legend(loc='upper right')
pypl.show()

ins = 2
hid1_units = 20
hid2_units = 10
outs = 2

Theta1 = np.random.randn(hid1_units,ins)
b1 = np.random.randn(m,hid1_units)

Theta2 = np.random.randn(hid2_units,hid1_units)
b2 = np.random.randn(m,hid2_units)

Theta3 = np.random.randn(outs,hid2_units)
b3 = np.random.randn(m,outs)


epochs = 10
batch_size = 100


for e in range(epochs):

	for b in range((int)(m/batch_size)):


		pass


	pass


























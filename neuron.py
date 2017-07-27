#  this whole part is needed to load the data from the CIFAR10 dataset
from __future__ import print_function
from six.moves import cPickle as pickle
import numpy as np
import os
from scipy.misc import imread
import platform
import time
import matplotlib.pyplot as plt
#import getch
plt.interactive(False)

print('All Packages imported!')

def load_pickle(f):
    version = platform.python_version_tuple()
    if version[0] == '2':
        return pickle.load(f)
    elif version[0] == '3':
        return pickle.load(f, encoding='latin1')
    raise ValueError("invalid python version: {}".format(version))


def load_CIFAR_batch(filename):
    """ load single batch of cifar """
    with open(filename, 'rb') as f:
        datadict = load_pickle(f)
        X = datadict['data']
        Y = datadict['labels']
        X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
        Y = np.array(Y)
        return X, Y


def load_CIFAR10(ROOT):
    """ load all of cifar """
    xs = []
    ys = []
    for b in range(1, 2):
        f = os.path.join(ROOT, 'data_batch_%d' % (b,))
        X, Y = load_CIFAR_batch(f)
        xs.append(X)
        ys.append(Y)
    Xtr = np.concatenate(xs)
    Ytr = np.concatenate(ys)
    del X, Y
    Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
    return Xtr, Ytr, Xte, Yte


def get_CIFAR10_data(num_training=49000, num_validation=1000, num_test=1000,
                     subtract_mean=True):
    """
    Load the CIFAR-10 dataset from disk and perform preprocessing to prepare
    it for classifiers. These are the same steps as we used for the SVM, but
    condensed to a single function.
    """
    # Load the raw CIFAR-10 data
    cifar10_dir = 'Cifar10data'
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

    # Subsample the data
    mask = list(range(num_training, num_training + num_validation))
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = list(range(num_training))
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = list(range(num_test))
    X_test = X_test[mask]
    y_test = y_test[mask]

    # Normalize the data: subtract the mean image
    if subtract_mean:
        mean_image = np.mean(X_train, axis=0)
        X_train -= mean_image
        X_val -= mean_image
        X_test -= mean_image

    # Transpose so that channels come first
    X_train = X_train.transpose(0, 3, 1, 2).copy()
    X_val = X_val.transpose(0, 3, 1, 2).copy()
    X_test = X_test.transpose(0, 3, 1, 2).copy()

    # Package data into a dictionary
    return {
        'X_train': X_train, 'y_train': y_train,
        'X_val': X_val, 'y_val': y_val,
        'X_test': X_test, 'y_test': y_test,
    }


def load_tiny_imagenet(path, dtype=np.float32, subtract_mean=True):
    """
    Load TinyImageNet. Each of TinyImageNet-100-A, TinyImageNet-100-B, and
    TinyImageNet-200 have the same directory structure, so this can be used
    to load any of them.

    Inputs:
    - path: String giving path to the directory to load.
    - dtype: numpy datatype used to load the data.
    - subtract_mean: Whether to subtract the mean training image.

    Returns: A dictionary with the following entries:
    - class_names: A list where class_names[i] is a list of strings giving the
      WordNet names for class i in the loaded dataset.
    - X_train: (N_tr, 3, 64, 64) array of training images
    - y_train: (N_tr,) array of training labels
    - X_val: (N_val, 3, 64, 64) array of validation images
    - y_val: (N_val,) array of validation labels
    - X_test: (N_test, 3, 64, 64) array of testing images.
    - y_test: (N_test,) array of test labels; if test labels are not available
      (such as in student code) then y_test will be None.
    - mean_image: (3, 64, 64) array giving mean training image
    """
    # First load wnids
    with open(os.path.join(path, 'wnids.txt'), 'r') as f:
        wnids = [x.strip() for x in f]

    # Map wnids to integer labels
    wnid_to_label = {wnid: i for i, wnid in enumerate(wnids)}

    # Use words.txt to get names for each class
    with open(os.path.join(path, 'words.txt'), 'r') as f:
        wnid_to_words = dict(line.split('\t') for line in f)
        for wnid, words in wnid_to_words.iteritems():
            wnid_to_words[wnid] = [w.strip() for w in words.split(',')]
    class_names = [wnid_to_words[wnid] for wnid in wnids]

    # Next load training data.
    X_train = []
    y_train = []
    for i, wnid in enumerate(wnids):
        if (i + 1) % 20 == 0:
            print('loading training data for synset %d / %d' % (i + 1, len(wnids)))
        # To figure out the filenames we need to open the boxes file
        boxes_file = os.path.join(path, 'train', wnid, '%s_boxes.txt' % wnid)
        with open(boxes_file, 'r') as f:
            filenames = [x.split('\t')[0] for x in f]
        num_images = len(filenames)

        X_train_block = np.zeros((num_images, 3, 64, 64), dtype=dtype)
        y_train_block = wnid_to_label[wnid] * np.ones(num_images, dtype=np.int64)
        for j, img_file in enumerate(filenames):
            img_file = os.path.join(path, 'train', wnid, 'images', img_file)
            img = imread(img_file)
            if img.ndim == 2:
                ## grayscale file
                img.shape = (64, 64, 1)
            X_train_block[j] = img.transpose(2, 0, 1)
        X_train.append(X_train_block)
        y_train.append(y_train_block)

    # We need to concatenate all training data
    X_train = np.concatenate(X_train, axis=0)
    y_train = np.concatenate(y_train, axis=0)

    # Next load validation data
    with open(os.path.join(path, 'val', 'val_annotations.txt'), 'r') as f:
        img_files = []
        val_wnids = []
        for line in f:
            img_file, wnid = line.split('\t')[:2]
            img_files.append(img_file)
            val_wnids.append(wnid)
        num_val = len(img_files)
        y_val = np.array([wnid_to_label[wnid] for wnid in val_wnids])
        X_val = np.zeros((num_val, 3, 64, 64), dtype=dtype)
        for i, img_file in enumerate(img_files):
            img_file = os.path.join(path, 'val', 'images', img_file)
            img = imread(img_file)
            if img.ndim == 2:
                img.shape = (64, 64, 1)
            X_val[i] = img.transpose(2, 0, 1)

    # Next load test images
    # Students won't have test labels, so we need to iterate over files in the
    # images directory.
    img_files = os.listdir(os.path.join(path, 'test', 'images'))
    X_test = np.zeros((len(img_files), 3, 64, 64), dtype=dtype)
    for i, img_file in enumerate(img_files):
        img_file = os.path.join(path, 'test', 'images', img_file)
        img = imread(img_file)
        if img.ndim == 2:
            img.shape = (64, 64, 1)
        X_test[i] = img.transpose(2, 0, 1)

    y_test = None
    y_test_file = os.path.join(path, 'test', 'test_annotations.txt')
    if os.path.isfile(y_test_file):
        with open(y_test_file, 'r') as f:
            img_file_to_wnid = {}
            for line in f:
                line = line.split('\t')
                img_file_to_wnid[line[0]] = line[1]
        y_test = [wnid_to_label[img_file_to_wnid[img_file]] for img_file in img_files]
        y_test = np.array(y_test)

    mean_image = X_train.mean(axis=0)
    if subtract_mean:
        X_train -= mean_image[None]
        X_val -= mean_image[None]
        X_test -= mean_image[None]

    return {
        'class_names': class_names,
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_val,
        'y_val': y_val,
        'X_test': X_test,
        'y_test': y_test,
        'class_names': class_names,
        'mean_image': mean_image,
    }


#############################################################
# load the data here
#######################################################

"""Begin by loading the data"""

m = 5
n = 3072
mval = 1000
mtest = 20

print('Loading data...')
data = get_CIFAR10_data(m,mval,mtest)
X = data['X_train']
y = data['y_train']
Xval = data['X_val']
yval = data['y_val']
Xtest = data['X_test']
ytest = data['y_test']

hid_layer1 = 1000
hid_layer2 = 500
out_units = np.max(y) + 1  # because 0 is the first class

# get smaller data sets for prototyping

X = X[:m, ]
y = y[:m, ]
Xval = Xval[:mval, ]
yval = yval[:mval, ]

# reshape some of the data matrices to use
X = np.reshape(X, (m, n))
Xval = np.reshape(Xval, (mval, n))  # max xtest and xval are 1000


###################################################################################################
# now do some feature scaling

X = X / np.max(X,axis=0,keepdims=True)
Xval = X / np.max(Xval,axis=0,keepdims=True)

# convert the ys to sparse Y matrices(one hot representation)
Y = np.zeros((m, out_units))  # ==> m*R = m*10
Y[np.arange(m), y] = 1

Yval = np.zeros((mval, out_units))  # ==> m*R = m*10
Yval[np.arange(mval), yval] = 1

print('data loaded successfully!\n')

############################################################
# Neural network begins here (its a ReLU network!)
# its got two layers

# define the activation routine

epsilon = 1e-4
np.random.seed(int(round(time.time())))

Theta1 = epsilon * np.random.randn(hid_layer1, n) / np.sqrt(hid_layer1)
b1 = np.random.rand(1,hid_layer1)

Theta2 = epsilon * np.random.randn(hid_layer2, hid_layer1) / np.sqrt(hid_layer2)
b2 = np.random.rand(1,hid_layer2)

Theta3 = epsilon * np.random.randn(out_units, hid_layer2) / np.sqrt(out_units)
b3 = np.random.rand(1,out_units)


def unit_step(X):
    X[X <= 0] = 0
    X[X > 0] = 1
    return X

def sigmoid(A):
    return 1 / (1 + np.exp(-A))

def R(A, B, bias):
    temp = A.dot(B.T) + bias
    #temp[temp <= 0] = 0

    return 1. / (1 + np.exp(-temp))


def dR(A, B, Prev):
    # returns derivative of R
    temp = A.dot(B.T).clip(min=0)
    temp = temp.clip(min=1, max=1)
    this = ((Prev * temp)[:, 1:]).dot(B)
    andthis = ((Prev * temp)[:, 1:]).T.dot(A)
    return this, andthis


def cost(A4):
    exp = np.exp(A4)
    exp /= np.max(exp,axis=None)
    prob = exp / np.sum(exp, axis=1)[:, None]
    costVector = -np.log(np.argmax(prob, axis=1))
    return prob, costVector


def gradientCheck(func, thisVariable, h = 1e-4):
    _, fprev = func(thisVariable+h)
    _, fnext = func(thisVariable-h)
    h = np.full(thisVariable.shape, h)
    # print(h.shape)
    return (1/((2*h)/(fprev-fnext)[:,None]))

iters = 2000
losses = []
learn_rate = 1e-3
reg = 1e-3
cost_training = []
cost_cross = []
dTheta1 = 0
dTheta2 = 0
dTheta3 = 0
db1 = 0
db2 = 0
db3 = 0

for k in range(iters):
    # c = getch.getch()
    # print(c)

    # Forward pass
    ###################################################################
    A2 = R(X, Theta1, b1)
    A3 = R(A2, Theta2, b2)
    A4 = A3.dot(Theta3.T) + b3

    exp = np.exp(A4)
    prob = exp / np.sum(exp,axis=1,keepdims=True)
    cost = np.sum(-np.log(prob[range(m), y])) + (np.sum(Theta1**2) + np.sum(Theta2**2)
                                           + np.sum(Theta3**2) + np.sum(b1**2) + np.sum(b2**2) + np.sum(b3**2))

    ###################################################################

    # see the loss on the prediction with current weights

    #cost = -1 / m * np.sum(Y * np.log(sigmoid(A4) + 1e-3) + (1 - Y) * np.log(1 - sigmoid(A4) + 1e-3)) + 1 / (2*m) * reg * \
     #       (np.sum(Theta1**2) + np.sum(Theta2**2) + np.sum(Theta3**2) + np.sum(b1**2) + np.sum(b2**2) + np.sum(b3**2))
    cost_training.append(cost)
    #if iters == 0:
    print('iteration = ', k, 'cost = ', cost)

    ###################################################################
    # backward pass begins here

    # dA4 = -1 / m * (Y * (1 - sigmoid(A4)))
    prob[range(m),y] -= 1
    dA4 = prob
    dA4 = A4 - Y

    dA3 = dA4.dot(Theta3) * (1 - np.power(A3, 2))
    dTheta3 = (A3.T).dot(dA4) + reg / m * Theta3.T
    db3 = np.sum(dA4, axis = 0, keepdims=True) + reg / m * b3

    dA2 = dA3.dot(Theta2) * (1 - np.power(A2, 2))
    dTheta2 = A2.T.dot(dA3) + reg / m * Theta2.T
    db2 = np.sum(dA3, axis = 0, keepdims=True) + reg / m * b2

    #dA1 = (dA2 * A2 * (1 - A2)).dot(Theta1)
    dTheta1 = X.T.dot(dA2) + reg / m * Theta1.T
    db1 = np.sum(dA2, axis = 0, keepdims=True) + reg / m * b1

    # update the weights and the biases
    Theta1 += -learn_rate * dTheta1.T
    Theta2 += -learn_rate * dTheta2.T
    Theta3 += -learn_rate * dTheta3.T

    b1 += -learn_rate * db1
    b2 += -learn_rate * db2
    b3 += -learn_rate * db3

    error = y != np.argmax(np.exp(A4) / np.sum(np.exp(A4),axis=1,keepdims=True),axis=1)
    error = error.sum() / float(error.size)
    print('Training error = ', error * 100, '% ', end = '')

    ###################################################################

# get training accuracy
A2 = R(X, Theta1, b1)
A3 = R(A2, Theta2, b2)
A4 = A3.dot(Theta3.T) + b3
exp = np.exp(A4)
prob = exp / np.sum(exp, axis=1, keepdims=True)
accuracy = np.mean(np.argmax(prob,axis=1) == y)
# print("for ",i," examples:")
print('Training accuracy = ', accuracy * 100, '%')

# get training accuracy
A2 = R(Xval, Theta1, b1)
A3 = R(A2, Theta2, b2)
A4 = A3.dot(Theta3.T) + b3
exp = np.exp(A4)
prob = exp / np.sum(exp, axis=1, keepdims=True)
accuracy = np.mean(np.argmax(prob,axis=1) == y)
print('Cross validation accuracy = ', accuracy * 100, '%\n')

# plt.plot(cost_cross,'b')
plt.figure()
plt.plot(cost_training)
plt.xlabel('# of iterations')
plt.ylabel('cost')
plt.draw()
plt.show()















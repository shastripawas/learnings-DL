import numpy as np
import math
from activations import sigmoid

def diff_sigmoid(ar_x):
    a = sigmoid(ar_x)
    return a*(np.ones(len(a))-a)

def diff_softmax(ar_x):
    sumexp_x = np.sum(np.exp(ar_x))
    return np.vectorize(lambda x:-((math.exp(x)/sumexp_x))**2)(ar_x)

def diff_loss(gt, pred):
    return pred-gt #((np.ones(len(gt))-gt)/(np.ones(len(pred))-pred))-(gt/pred)


import math
import numpy as np

def sigmoid(ar_x):
    ar_x = np.clip(ar_x, -500, 500) 
    return  1/(1+np.exp(-ar_x))

def tanh(x):
    return (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))

def softmax(ar_x):
    sumexp_x = np.sum(np.exp(ar_x))
    return np.vectorize(lambda x:(math.exp(x)/sumexp_x))(ar_x)

def fire(x, activation):
    if activation=='sigmoid':
        return sigmoid(x)
    elif activation=='tanh':
        return tanh(x)
    else:
        return x
    

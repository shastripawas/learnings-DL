import numpy as np

def cross_entropy(gt, pred):
    return -((gt*np.log(pred))) #+ ((np.ones(len(gt))-gt)*np.log(1-pred)))


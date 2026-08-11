import cupy as np

def cross_entropy(gt, pred):
    #print(f"inside loss gtdim : {np.shape(gt)}, pred dim : {np.shape(pred)}")
    return -((gt*np.log(pred))) #+ ((np.ones(len(gt))-gt)*np.log(1-pred)))


from fpass import forward_pass
from bpass import bpass, update
from loss import cross_entropy
from activations import softmax
import cupy as np

data = np.random.rand(12)*100
config = [3,4,3]
lays = [len(data)]+config
W = [np.random.rand(lays[i+1], lays[i]+1) for i in range(len(lays)-1)]
gt = [1, 0, 0]
fp_dict = forward_pass(data, config, 'sigmoid', W)
loss = cross_entropy(gt, softmax(fp_dict['a'][-1]))
grads = bpass(gt, fp_dict, W)
print(f"grads : {grads}")

W_new = update(W, 1, grads)
# print(f"Forward pass output : {forward_pass(data, config, 'sigmoid', W)}")
# print(f"Forward pass output : {bpass(data, config, 'sigmoid', W)}")
#print(f"wneww : {W==W_new}")
print(f"W_init : {W}")
print(f"W_new : {W_new}")
print(f"loss : {loss}")
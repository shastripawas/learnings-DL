import numpy as np
from activations import fire, softmax

def build_layer_forward(n_curr, n_prev, out_prev, activation, W):
    #W = np.random.rand(n_curr, n_prev+1)
    out_prev = np.array(list(out_prev)+[1])
    # print(f"w_shape : {np.shape(W)}")
    # print(f"outprev_shape : {np.shape(out_prev)}")
    out_un = W.dot(out_prev)
    out = fire(out_un, activation)
    return out_un, out

def forward_pass(data, config, activation, W):
    fp_dict = {'a' : [], 'z': []}
    out = data
    n_prev = len(data)
    for i in range(len(config)):
        fp_dict['a'].append(out)
        n_curr = config[i]
        out_un, out = build_layer_forward(n_curr, n_prev, out, activation, W[i]) 
        # print(f"out_len : {len(out)}")
        #print("out_un len : ")
        fp_dict['z'].append(out_un)
        n_prev = n_curr
    fp_dict['a'].append(out_un)
    return fp_dict





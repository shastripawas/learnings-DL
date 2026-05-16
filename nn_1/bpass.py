import numpy as np
from compute_gradient import diff_loss, diff_sigmoid, diff_softmax
from activations import softmax

def bpass(gt, fp_dict, W):
    Pr = softmax(fp_dict['a'][-1])
    grads = []
    # grad_loss = diff_loss(gt,Pr)
    # grad_soft = diff_softmax(fp_dict['a'][-1])
    # grad_sig_init = diff_sigmoid(fp_dict['z'][-1])
    fin_grad_init = np.array(np.append([fp_dict['a'][-2]],1)*len(fp_dict['z'][-1]))
    #fin_grad_init = fp_dict['a'][-2]
    #print(fin_grad_init)
    #print(grad_sig_init)
    # print(f"fp_dict : {fp_dict['a']}")
    # print(f"len z : {len(fp_dict['z'])}")

    multiplier_prev = (Pr-gt)[:, None]#(grad_soft*grad_loss*grad_sig_init)[:, None]
    grad_prev = multiplier_prev*fin_grad_init
    grads.append(grad_prev)
    for i in range(len(fp_dict['z'])-2, -1,-1):
        # print(f"z_shapes : {np.shape(fp_dict['z'][i])}")
        # print(f"i-1 : {i-1}")
        # print(f"prev act_shapes : {np.shape(fp_dict['a'][i])}")
        # print(f"multipler_prev_shape : {np.shape(multiplier_prev)}")
        # print(f"weight_shape : {np.shape(W[i+1])}")
        grad_sig = diff_sigmoid(fp_dict['z'][i])
        fin_grad_temp = np.array(np.append([fp_dict['a'][i]],1)*len(fp_dict['z'][i]))
        temp = multiplier_prev*W[i+1][:,:-1]
        multiplier_new = (np.sum(temp, axis=0)*grad_sig)[:, None]
        # print(f"multiplier_new_shape : {np.shape(multiplier_new)}")
        # print(np.shape(fin_grad))

        grad = fin_grad_temp*multiplier_new
        grads.append(grad)
        multiplier_prev = multiplier_new
    
    return grads

def update(W, lr, grads):
    W_upd = W.copy()
    n=len(W)
    for i in range(n):
        W_upd[i] = W[i]-(lr*grads[n-1-i])
        # print(f"grad : {(lr*grads[n-1-i])}")
        # print(f"is_equal : {W_upd[i]==W[i]}")
        
    return W_upd

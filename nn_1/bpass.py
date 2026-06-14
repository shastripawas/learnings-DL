import numpy as np
from compute_gradient import diff_loss, diff_sigmoid, diff_softmax
from activations import softmax

def bpass(gt, fp_dict, W):
    Pr = softmax(fp_dict['a'][-1])
    grads = []
    # grad_loss = diff_loss(gt,Pr)
    # grad_soft = diff_softmax(fp_dict['a'][-1])
    # grad_sig_init = diff_sigmoid(fp_dict['z'][-1])
    acts_prev_l = fp_dict['a'][-2]
    acts_prev_l = np.hstack([acts_prev_l, np.ones((acts_prev_l.shape[0], 1))])
    #fin_grad_init = np.array(np.append([fp_dict['a'][-2]],1)*len(fp_dict['z'][-1]))
    fin_grad_init = np.repeat(acts_prev_l[:, None, :], np.shape(fp_dict['z'][-1])[1], axis=1)
    #fin_grad_init = fp_dict['a'][-2]
    # print(f"fin_grad_init shape : {np.shape(fin_grad_init)}")
    #print(grad_sig_init)
    # print(f"fp_dict : {fp_dict['a']}")
    # print(f"len z : {len(fp_dict['z'])}")

    multiplier_prev = (Pr-gt)#[:,:, None]#(grad_soft*grad_loss*grad_sig_init)[:, None]
    #print(f"multiplier_prev shape : {np.shape(multiplier_prev)}")
    grad_prev = fin_grad_init * multiplier_prev[:,:,None]
    # print(f"grad_prev shape : {np.shape(grad_prev)}")
    # print(f"wlast shape : {np.shape(W[len(W)-1])}")
    grads.append(grad_prev.mean(axis=0))
    for i in range(len(fp_dict['z'])-2, -1,-1):
        # print(f"z_shapes : {np.shape(fp_dict['z'][i])}")
        # print(f"i-1 : {i-1}")
        # print(f"prev act_shapes : {np.shape(fp_dict['a'][i])}")
        # print(f"multipler_prev_shape : {np.shape(multiplier_prev)}")
        # print(f"weight_shape : {np.shape(W[i+1])}")
        grad_sig = diff_sigmoid(fp_dict['z'][i])
        acts_prev_temp = fp_dict['a'][i]
        acts_prev_temp = np.hstack([acts_prev_temp, np.ones((acts_prev_temp.shape[0], 1))])
        fin_grad_temp = np.repeat(acts_prev_temp[:, None, :], np.shape(fp_dict['z'][i])[1], axis=1)
        #fin_grad_temp = np.array(np.append([fp_dict['a'][i]],1)*len(fp_dict['z'][i]))
        # print(f"fin_grad_temp shape : {np.shape(fin_grad_temp)}")
        # print(f"multiplier_prev shape : {np.shape(multiplier_prev)}")
        # print(f"grad sig shape : {np.shape(grad_sig)}")
        temp = multiplier_prev.dot(W[i+1][:,:-1])
        multiplier_new = temp * grad_sig
        #multiplier_new = (np.sum(temp, axis=0)*grad_sig)[:, None]
        # print(f"multiplier_new_shape : {np.shape(multiplier_new)}")
        # print(np.shape(fin_grad))

        grad = fin_grad_temp*multiplier_new[:,:,None]
        grads.append(grad.mean(axis=0))
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

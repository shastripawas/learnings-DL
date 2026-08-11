from sklearn.datasets import fetch_openml
import cupy as np
import numpy as npo
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from fpass import forward_pass
from bpass import bpass, update
from loss import cross_entropy
from activations import softmax
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle 
import time

mnist = fetch_openml('mnist_784', version=1)

X = mnist.data   # shape: (70000, 784)
y = mnist.target
y = npo.array([int(i) for i in list(y)])
gt=[]
for i in y:
    #print(i)
    #print(type(i))
    temp = [1 if j==i else 0 for j in range(10)]
    temp=npo.array(temp)
    gt.append(temp)
gt = npo.array(gt)
X_train, X_test, y_train, y_test = train_test_split(
    X, gt,
    test_size=0.1,      # 80-20 split
    random_state=42,    # reproducibility
    stratify=gt          # keeps class balance
)
X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)
X_train = X_train / 255.0
X_test = X_test / 255.0
#print(f"y0 : {y}")
# print(type(np.array(X_train)))

# print(f"y_shape : {y[:10]}")
# print(f"ytrain_shape : {np.shape(y_train)}")

config = [64,10]
lays = [len(X_train[0])]+config
#W = [np.random.rand(lays[i+1], lays[i]+1)*0.01 for i in range(len(lays)-1)]
W = [
    np.random.randn(lays[i+1], lays[i] + 1) * np.sqrt(2 / lays[i])
    for i in range(len(lays) - 1)
]
num_iters = 25
it = 0
lr=0.01
#grads_ = [np.zeros(np.shape(i)) for i in W]
#print(f"grads_av : {grads_av}")

loss_lst = []
batch_size = 512
start = time.perf_counter()
print("Starting Training....")
while it<num_iters:
    m=len(X_train)
    grads_sum = [np.zeros(np.shape(W[len(W)-1-i])) for i in range(len(W))]
    los_sum=0
    i=0
    count=0
    while i<m:
        data = X_train[i:min(i+batch_size,m), :]
        gt = y_train[i:min(i+batch_size,m), :]
        # print(f"ytrain dim : {np.shape(y_train)}")
        # print(f"xtrain0 dim : {np.shape(y_train[0].T)}")
        #print(f"data dim : {np.shape(data)}")
        fp_dict = forward_pass(data, config, 'sigmoid', W)
        # print("forward pass done!")
        loss = cross_entropy(gt, np.clip(softmax(fp_dict['a'][-1]), 1e-9, 1 - 1e-9))
        # print(np.shape(loss))
        # print(f"loss : {loss}")
        # print(f"mean_loss_vec : {loss.mean(axis=0)}")
        av_los = loss.mean(axis=0).sum()
        los_sum+=av_los
        grads = bpass(gt, fp_dict, W)
        # grads_sum = [grads_sum[i] + grads[i] for i in range(len(grads))]
        #print("train loop ran successfully!")
        W = update(W, lr, grads)
        count+=1
        i+=batch_size
        #print(f"batch : {count}, loss : {av_los}")

    # grads_av = [i/m for i in grads_sum]
    # W = update(W, lr, grads_av)
    #print(f"count : {count}")
    loss_lst.append(los_sum/count)
    #print(f"epoch : {it+1}, loss_vec : {los_sum/m}, average loss : {np.sum(los_sum/m)}")
    print(f"epoch : {it+1}, average loss : {los_sum/count}")
    it+=1
end = time.perf_counter()
print(f"Training loop ran for {end-start} seconds")
with open('models/epo_25_lr_0.01_mb_64-10.pkl', 'wb') as f:
    pickle.dump(W, f)
plt.plot([j.item() for j in loss_lst])
plt.show()

print("Model Trained. Running inference on test set......")
tn = len(X_test)
pred_nums=[]
y_test_nums = []

# with open("models/epo_25_lr_0.01_mb_64-10.pkl", "rb") as f:
#     W = pickle.load(f)

data_prev = np.zeros(len(X_test[0]))
for tex in range(tn):
    data = X_test[[tex]]
    #print(f"data : {data}")
    #print(f"is_dat_equal : {data_prev==data}")
    gt = y_test[[tex]]
    fp_dict = forward_pass(data, config, 'sigmoid', W)
    pred = softmax(fp_dict['a'][-1])
    #print(f"pred : {pred}")
    pred_num = np.argmax(pred)
    act_num = np.argmax(gt)
    pred_nums.append(pred_num)
    y_test_nums.append(act_num)
    data_prev=data

pred_nums = np.array(pred_nums)
y_test_nums = np.array(y_test_nums)
print(f"un_nums : {np.unique(pred_nums)}")
print(f"original : {y_test_nums[:10]}, predicted : {pred_nums[:10]}")
# print(type(y_test_nums))
# print(type(y_test_nums[0]))
#y_test_nums = np.array([int(i) for i in list(y_test)])
print(f"Accuracy : {sum(pred_num==y_test_nums)/len(y_test_nums)}")
y_test_nums = np.asnumpy(y_test_nums)
pred_nums = np.asnumpy(pred_nums)
print(f"Classification Report : {classification_report(y_test_nums, pred_nums)}")








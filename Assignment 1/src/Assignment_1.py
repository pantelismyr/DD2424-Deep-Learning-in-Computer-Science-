"""
@author = Pantelis
"""
import pickle as cPickle
import numpy as np
from timeit import default_timer
import matplotlib.pyplot as plt


# Global Var
d = 3072      # image dimention
K = 10        # total number of labels
N = 10000     # total number of images
mu = 0        
sigma = .01
mode = 'SVM'                 # change to 'SVM'
drop_rate = False          # change to True to drop the value of eta in every epoch
lamda = 0.0001


class GD_params:
    #GDparams is an object containing the parameter values n batch, eta and n epochs
    def __init__(self, n_batch, eta, n_epochs):
        self.n_batch = n_batch
        self.eta = eta
        self.n_epochs = n_epochs

    def __str__(self):
        return "GD parameters are: " + "n_batch = " + str(self.n_batch)  +  ", " "eta = "  \
                + str(self.eta) + ", "+ "n_epochs = "  + str(self.n_epochs)

def load_data(filename):
    # Load dataset
    X = np.array;
    with open(filename, 'rb') as fo:
        dict = cPickle.load(fo, encoding='bytes')
    X = dict[b'data'].T / 255;
    y = dict[b'labels'];# list of labels as int
    Y = np.zeros((10, X.shape[1]));
    for i in range(len(y)):
        Y[y[i], i ] = 1;   
    return X, Y, y
def model_params():
    # Initialize model parameters
    np.random.seed(400)
    W = np.random.normal(mu, sigma, (K, d))
    b = np.random.normal(mu, sigma, (K, 1))
    return W, b

def evaluate_classifier(X, W, b):
    s = np.dot(W, X) + b
    if mode == 'softmax':
        return np.exp(s) / np.sum(np.exp(s), axis=0)
    elif mode == 'SVM':
        return s 

def SVM_loss(s, label):
    loss = 0.0
    for i in range(len(s)):
        if i != label:
            loss += np.maximum(0, s[i] - s[label] + 1)   
    return loss          



# compute the cost function
def compute_cost(X, Y, label, W, b, lamda, flag = True):
    if mode == 'softmax':
        if flag:
            # when Y is encoded by a one-hot representation
            l_cross = np.diag(-np.log(np.dot(Y.T, evaluate_classifier(X, W, b))))
        else:
            # otherwise
            l_cross = np.diag(-np.log(evaluate_classifier(X, W, b)))
    elif mode == 'SVM':
        l_cross = 0
        for i in range(len(label)):
            x = np.expand_dims(X[:,i], axis = 1)
            s = np.dot(W, x) + b
            l_cross += SVM_loss(s, label[i])
    regularization = 2 * lamda*np.sum(W**2)
    loss = l_cross.sum()
    cost = loss / len(label) + regularization
    return cost, loss

def compute_accuracy(X, y, W, b):
    pred = np.argmax(evaluate_classifier(X, W, b), axis = 0)
    eval_out = np.abs(pred - np.argmax(y, axis = 0))
    error = np.count_nonzero(eval_out) / X.shape[1]
    return 1 - error
        

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ Compute Gradients ---------------->>
def compute_gradients(X, Y, y_tr, P, W, b, lamda):
    # compute gradients analytically 
    grad_W = np.zeros((K, d))
    grad_b = np.zeros((K, 1))
    diag_p = np.zeros((K, K))
    if mode == 'softmax':
        for i in range(X.shape[1]):                
            y = np.copy(Y[:, i])
            y = y.reshape(K,1)
            y_T = y.T
            p = np.copy(P[:, i])
            p = p.reshape(K, 1)
            p_T = p.T
            np.fill_diagonal(diag_p, p)
            x = np.copy(X[:, i])
            x = x.reshape(d, 1)
            g = -(np.dot(y_T, (diag_p - np.dot(p, p_T)))) / np.dot(y_T, p)
            grad_b += g.T
            grad_W += np.dot(g.T, x.T)
        grad_W = grad_W / X.shape[1]
        grad_b = grad_b / X.shape[1]  
        J_grad_W = grad_W + 2 * lamda * W
        J_grad_b = grad_b              
        return J_grad_W, J_grad_b
    elif mode == 'SVM':
        for i in range(X.shape[1]): 
            x = np.expand_dims(X[:, i], axis = 1)
            s = (np.dot(W, x) + b).clip(0)
            g = (s[y_tr[i]] - s - 1 < 0) * 1
            g[y_tr[i]] = - (np.sum(g) - 1)
            grad_b += g
            grad_W += np.dot(g, x.T)
        grad_W /= len(y_tr)
        grad_b /= len(y_tr)
        J_grad_W = grad_W + lamda * W
        J_grad_b = grad_b  
        return J_grad_W, J_grad_b        
   

def compute_grads_num_slow(X, Y, W, b, lamda, h = 1e-6):
    # compute gradients numerically
    grad_W_num = np.zeros((K, d))
    grad_b_num = np.zeros((K, 1))
    for i in range(len(b)):
        b_try = np.copy(b)      
        b_try[i] -= h       
        c1 = compute_cost(X, Y, None, W, b_try, lamda)[0]
        b_try = np.copy(b)
        b_try[i] += h
        c2 = compute_cost(X, Y, None, W, b_try, lamda)[0]
        grad_b_num[i] = (c2-c1) / (2 * h)
    for i in range(len(W)):
        for j in range(len(W[0])):
            W_try = np.copy(W)
            W_try[i][j] -= h
            c1 = compute_cost(X, Y, None, W_try, b, lamda)[0]
            W_try = np.copy(W)
            W_try[i][j] += h
            c2 = compute_cost(X, Y, None, W_try, b, lamda)[0]
            grad_W_num[i][j] = (c2-c1) / (2*h)
    return grad_b_num, grad_W_num 
    

def check_grad(grad_b, grad_W, grad_b_num, grad_W_num, W, flag):
    # Remember lamda = 0, so grad_b = J_grad_b and grad_W = J_grad_W 
    W_abs_diff = np.absolute(grad_W - grad_W_num)
    b_abs_diff = np.absolute(grad_b - grad_b_num)
    diff_W_average = np.average(W_abs_diff)
    if flag == 1:
        # When the gradient vectors have small values this approach may fail!
        if diff_W_average < 1e-5:
            return "Accepted! The absolute difference is less than 1e-6. (flag == 1)"
        else:
            print("average(W_abs_diff) =", diff_W_average)
            return "Warning...!" 
    elif flag ==2:
        # More reliable method
        grad_W_abs = np.absolute(grad_W)
        grad_W_num_abs = np.absolute(grad_W_num)
        sum_W = grad_W_abs + grad_W_num_abs
        res = diff_W_average / np.amax(sum_W) 
        if res < 1e-5:
            print("res =", res)
            return "Accepted! The absolute difference is less than 1e-6. (flag == 2)"
        else:
            print("average(W_abs_diff) =", res)
            return "Warning...!"                
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@-------------------------------<<

# ----------------------------------------------- Mini-Batch ------------------------------------------->>
def mini_batch(X, Y, y_tr, X_val, Y_val_hot, y_val, GDparams, W, b, lamda):     
    J_train_ls = []
    J_val_ls = []
    loss_train_ls = []
    loss_val_ls = []
    acc_train_ls = []
    acc_val_ls = []
    W_ls = []
    b_ls = []

    #Initial 
    J_train, loss_train = compute_cost(X, Y, y_tr, W, b, lamda)
    J_val, loss_val = compute_cost(X_val, Y_val_hot, y_val, W, b, lamda)
    J_train_ls.append(J_train)
    J_val_ls.append(J_val)
    loss_train_ls.append(loss_train.sum() / X.shape[1])
    loss_val_ls.append(loss_val.sum() / X_val.shape[1])  
    acc_train_ls.append(compute_accuracy(X, Y, W, b))
    acc_val_ls.append(compute_accuracy(X_val, Y_val_hot, W, b))
    
    for epoch in range(GDparams.n_epochs):
        # if mode == 'softmax':
        #     for j in range(1, N // GDparams.n_batch):
        #         # set of mini-batches for 1 epoch
        #         j_start = (j - 1) * GDparams.n_batch + 1
        #         j_end = j * GDparams.n_batch
        #         X_batch = X[:, j_start : j_end]
        #         Y_batch = Y[:, j_start : j_end]
        #         y_batch_tr = y_tr[j_start : j_end]   
        #         y_batch_val = y_val[j_start : j_end]
        #         if mode == 'softmax':
        #             J_grad_W, J_grad_b = compute_gradients(X_batch, Y_batch, None, evaluate_classifier(X_batch, W, b), W, b, lamda)
        #         elif mode == 'SVM':
        #             J_grad_W, J_grad_b = compute_gradients(X_batch, Y_batch, y_batch_tr, None, W, b, lamda)
        #         W -= GDparams.eta * J_grad_W
        #         b -= GDparams.eta * J_grad_b
        # elif mode == 'SVM': 
        i = 0
        while i < len(y_tr):
            if i + GDparams.n_batch > len(y_tr):
                X_batch = X[:, i:len(y_tr)]
            else:
                X_batch = X[:, i:i + GDparams.n_batch]
            if i + GDparams.n_batch > len(y_tr):
                Y_batch = Y[:, i:len(y_tr)]
            else:
                Y_batch = Y[:, i:i + GDparams.n_batch] 
            if i + GDparams.n_batch > len(y_tr):
                y_batch_tr = y_tr[i:len(y_tr)]
            else:
                y_batch_tr = y_tr[i:i + GDparams.n_batch] 
            
            if mode == 'softmax':
                J_grad_W, J_grad_b = compute_gradients(X_batch, Y_batch, None, evaluate_classifier(X_batch, W, b), W, b, lamda)
            elif mode == 'SVM':
                J_grad_W, J_grad_b = compute_gradients(X_batch, Y_batch, y_batch_tr, None, W, b, lamda)
            W -= GDparams.eta * J_grad_W
            b -= GDparams.eta * J_grad_b
            i += GDparams.n_batch
                
        if drop_rate == True:
            # Play around with decaying the learning rate by a factor ~ .9 after each epoch. (Ex. 2(d))
            GDparams.eta -= GDparams.eta - GDparams.eta * 0.9     
        
        J_train, loss_train = compute_cost(X, Y, y_tr, W, b, lamda)
        J_val, loss_val = compute_cost(X_val, Y_val_hot, y_val, W, b, lamda)    
        
        #-- cost function
        J_train_ls.append(J_train)
        J_val_ls.append(J_val)

        #-- error
        loss_train_ls.append(loss_train.sum() / X.shape[1])
        loss_val_ls.append(loss_val.sum() / X_val.shape[1])

        #-- accuracy
        acc_train_ls.append(compute_accuracy(X, Y, W, b))
        acc_val_ls.append(compute_accuracy(X_val, Y_val_hot, W, b))
        max_val_acc = np.max(acc_val_ls)
        
        #weight list
        W_ls.append(W)
        b_ls.append(b)

        print("Epoch", epoch)
    best = np.argmax(acc_val_ls)    
    return J_train_ls, J_val_ls, loss_train_ls, loss_val_ls, acc_train_ls, acc_val_ls, W_ls[best - 1], max_val_acc




def graph_vis(J_train_ls, J_val_ls, loss_train_ls, loss_val_ls, acc_train_ls, acc_val_ls, GDparams, max_val_acc):
    #--- Plots
    fig0 = plt.figure(figsize=(20,10))
    fig0.add_subplot(2, 3, 1)
    l1 = plt.plot(J_train_ls,'g', label = 'Training loss')
    l2 = plt.plot(J_val_ls,'r', label = 'Validation loss')
    plt.legend(loc = 'upper right')
    plt.ylabel('loss')
    plt.xlabel('epochs')
    plt.title('Cost function vs Epochs')

    fig0.add_subplot(2, 3, 2)
    l3 = plt.plot(loss_train_ls,'g', label = 'Total Training loss')
    l4 = plt.plot(loss_val_ls,'r', label = 'Total Validation loss')
    plt.legend(loc = 'upper right')
    plt.ylabel('loss')
    plt.xlabel('epochs')
    plt.title('Loss vs Epochs')

    fig0.add_subplot(2, 3, 3)
    l5 = plt.plot(acc_train_ls,'g', label = 'Training accuracy')
    l6 = plt.plot(acc_val_ls,'r', label = 'Validation accuracy')
    plt.legend(loc = 'upper left')
    plt.ylabel('accuracy')
    plt.xlabel('epochs')
    plt.title('The maximum accuracy of the validation set is: ' + str(max_val_acc * 100) + '%')
    plt.savefig("../figs/graphs_l=" + str(lamda) + "_b=" + str(GDparams.n_batch) + "_eta=" \
                + str(GDparams.eta) + "_ep=" + str(GDparams.n_epochs) + ".png", bbox_inches='tight')
   

def weights_vis(W, GDparams):
    # weight visualization
    images = []
    for img in W:
        raw_img=np.rot90(img.reshape(3, 32, 32).T, -1)
        image = ((raw_img-np.min(raw_img))/(np.max(raw_img)-np.min(raw_img)))
        images.append(image)
    fig1 = plt.figure(figsize = (20, 5))
    for idx in range(len(W)):
        ax = fig1.add_subplot(1, 10, idx+1) 
        ax.set_title('Class %s'%(idx+1))
        ax.imshow(images[idx])
        ax.axis('off')
    plt.savefig("../figs/weights_l=" + str(lamda) + "_b=" + str(GDparams.n_batch) + "_eta=" \
                + str(GDparams.eta) + "_ep=" + str(GDparams.n_epochs) + ".png", bbox_inches='tight')                        
   


def main():
     # Load dataset
    print("Loading data ...")
    X_train, Y_train_hot, y_train = load_data('../Datasets/cifar-10-batches-py/data_batch_1')
    X_val, Y_val_hot, y_val = load_data('../Datasets/cifar-10-batches-py/data_batch_2')
    #X_test, Y_test_hot = load_data('../Datasets/cifar-10-batches-py/test_batch')

    # # Decrease the size of the validation set down to ~1000
    # X_train_new = np.concatenate((X_train.T, X_val.T[:len(X_val.T) - 1000])).T
    # Y_train_hot_new = np.concatenate((Y_train_hot.T,Y_val_hot.T[:len(Y_val_hot.T) - 1000])).T
    # y_train_new = np.concatenate((y_train,y_val[:len(X_val.T) - 1000]))
    # X_val_new = X_val[:,len(X_val.T) - 1000:]
    # Y_val_hot_new = Y_val_hot[:, len(X_val.T) - 1000:]
    # y_val_new = y_val[len(X_val.T) - 1000:]
    print("Done!") 

    
    
    # Initialize model params
    W, b = model_params()

    #--- Mini-Batch
    GDparams = GD_params(n_batch = 100, eta = 0.001 , n_epochs = 200)
    start = default_timer()
    # Exercise 1:
    #J_train_ls, J_val_ls, loss_train_ls, loss_val_ls, acc_train_ls, acc_val_ls, W_star, max_val_acc = mini_batch(X_train, Y_train_hot, y_train, X_val, Y_val_hot, y_val, GDparams, W, b, lamda)
    
    #--- Exercise 2:
    # 1.(a) Decrease the size of the validation set down to ~1000
    #J_train_ls, J_val_ls, loss_train_ls, loss_val_ls, acc_train_ls, acc_val_ls, W_star, max_val_acc = mini_batch(X_train_new ,  Y_train_hot_new , y_train_new , X_val_new, Y_val_hot_new, y_val_new, GDparams, W, b, lamda)
    # 1.(d):        *****Turn global 'drop_rate' to "True"*****
    #J_train_ls, J_val_ls, loss_train_ls, loss_val_ls, acc_train_ls, acc_val_ls, W_star, max_val_acc = mini_batch(X_train, Y_train_hot, y_train, X_val, Y_val_hot, y_val, GDparams, W, b, lamda)
    # 2. Train network by minimizing the SVM multi-class loss       *****Turn global 'mode' to 'SVM'*****
    J_train_ls, J_val_ls, loss_train_ls, loss_val_ls, acc_train_ls, acc_val_ls, W_star, max_val_acc = mini_batch(X_train, Y_train_hot, y_train, X_val, Y_val_hot, y_val, GDparams, W, b, lamda)


    print("time in sec: ", str(default_timer() - start))
    
    #--- Visualize Weights
    graph_vis(J_train_ls, J_val_ls, loss_train_ls, loss_val_ls, acc_train_ls, acc_val_ls, GDparams, max_val_acc)
    weights_vis(W_star, GDparams)
    plt.show()

if __name__ == "__main__":
    main()

    
   
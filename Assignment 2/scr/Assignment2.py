"""
@author = Pantelis
"""

import pickle as cPickle
import numpy as np
from timeit import default_timer
import matplotlib.pyplot as plt
import numpy.matlib
np.set_printoptions(formatter={'float_kind':lambda x: "%.5f" % x})

'''--- Global Var ---'''
m = 100        # nodes in the hidden layer
d = 3072      # image dimention
K = 10        # total number of labels
N = 10000     # total number of images
mu = 0        
sigma = .001
mode = 'ReLU'
drop_out = True

class GD_params:
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
    X = dict[b'data'].T / 255

    y = dict[b'labels'];# list of labels as int
    Y = np.zeros((10, X.shape[1]));
    for i in range(len(y)):
        Y[y[i], i ] = 1;   
    return X, Y, y
def model_params():
    # Initialize model parameters
    np.random.seed(400)
    W_1 = np.random.normal(mu, sigma, (m, d))
    W_2 = np.random.normal(mu, sigma, (K, m))
    b_1 = np.zeros((m, 1))
    b_2 = np.zeros((K, 1))
    mod_params = [W_1, W_2, b_1, b_2]
    return mod_params

def dropout_evaluate_classifier(X, W_1, W_2, b_1, b_2):
    s_1 = np.dot(W_1, X) + b_1
    h = activation_func(s_1)
    drop = np.random.choice([0, 1], size=(h.shape[0],h.shape[1]), p=[0.1, 0.9])
    h = np.multiply(h, drop)
    s = np.dot(W_2, h) + b_2
    softmax = np.exp(s) / np.sum(np.exp(s), axis = 0, keepdims = True)
    return softmax, h


def evaluate_classifier(X, W_1, W_2, b_1, b_2):
    s_1 = np.dot(W_1, X) + b_1
    h = activation_func(s_1)
    s = np.dot(W_2, h) + b_2
    softmax = np.exp(s) / np.sum(np.exp(s), axis = 0, keepdims = True)
    return softmax, h
    
def activation_func(s):
    if mode == 'ReLU':
        h = np.maximum(0, s)
        return h
    elif mode == 'LeakyReLU':
        h = np.maximum(.01 * s, s)
        return h   

def compute_cost(X, Y, W_1, W_2, b_1, b_2, lamda):
    P, h = evaluate_classifier(X, W_1, W_2, b_1, b_2)
    # when Y is encoded by a one-hot representation then l_cross=
    l_cross = -np.log(np.diag(np.dot(Y.T, P)))
    regularization = lamda * (np.sum(W_1 ** 2) + np.sum(W_2 ** 2))
    loss = l_cross.sum() 
    cost = loss / X.shape[1]  + regularization
    return cost, loss

def compute_accuracy(X, Y, W_1, W_2, b_1, b_2):
    P, h = evaluate_classifier(X, W_1, W_2, b_1, b_2)
    pred = np.argmax(P, axis = 0)
    eval_out = np.abs(pred - np.argmax(Y, axis = 0))
    error = np.count_nonzero(eval_out) / X.shape[1]
    return 1 - error    
# -------------------------------------------------------------------------------------------------------- Compute Gradients ----->>>
def compute_gradients(x, y, p, h, W_1, W_2, b_1, b_2, lamda):
    '''
    compute gradients analytically 
    '''
    # Set all entries to zero
    grad_W_1 = np.zeros((m, d))
    grad_W_2 = np.zeros((K, m))
    grad_b_1 = np.zeros((m, 1))
    grad_b_2 = np.zeros((K, 1))

    g = p - y
    grad_b_2 = g.sum()
    grad_W_2 = np.dot(g, h.T) 
    
    # Propagate the gradients
    g = np.dot(g.T, W_2)
    s_1 = np.dot(W_1, x) + b_1
    if mode == 'ReLU':
        ind = 1 * (s_1 > 0)
        g = g.T * ind
    elif mode == 'LeakyReLU':
        ind = (1 * (s_1 > 0) + 0.01 * (s_1 < 0))
        g = g.T * ind
    # first layer parameters computed
    grad_b_1 = g.T.sum()
    grad_W_1 = np.dot(g, x.T)


    grad_W_1 = grad_W_1 / x.shape[1]
    grad_W_2 = grad_W_2 / x.shape[1]
    grad_b_1 = grad_b_1 / x.shape[1] 
    grad_b_2 = grad_b_2 / x.shape[1]  

    # Add the gradient for the regularization term
    J_grad_W_1 = grad_W_1 + 2 * lamda * W_1
    J_grad_W_2 = grad_W_2 + 2 * lamda * W_2
    J_grad_b_1 = grad_b_1
    J_grad_b_2 = grad_b_2

    return J_grad_W_1, J_grad_W_2, J_grad_b_1, J_grad_b_2
    
def compute_grads_num_slow(X, Y, W_1, W_2, b_1, b_2, lamda, h = 1e-5):
    # compute gradients numerically
    grad_W_1_num = np.zeros((m, d))
    grad_W_2_num = np.zeros((K, m))
    grad_b_1_num = np.zeros((m, 1))
    grad_b_2_num = np.zeros((K, 1))
    for i in range(b_1.shape[1]):
        b_1_try = np.copy(b_1)      
        b_1_try[i] -= h       
        c1 = compute_cost(X, Y, W_1, W_2, b_1_try, b_2, lamda)[0]
        b_1_try = np.copy(b_1)
        b_1_try[i] += h
        c2 = compute_cost(X, Y, W_1, W_2, b_1_try, b_2, lamda)[0]
        grad_b_1_num[i] = (c2-c1) / (2 * h)
    for i in range(b_2.shape[1]):
        b_2_try = np.copy(b_2)      
        b_2_try[i] -= h       
        c1 = compute_cost(X, Y, W_1, W_2, b_1, b_2_try, lamda)[0]
        b_2_try = np.copy(b_2)
        b_2_try[i] += h
        c2 = compute_cost(X, Y, W_1, W_2, b_1, b_2_try, lamda)[0]
        grad_b_2_num[i] = (c2-c1) / (2 * h)    
    
    for i in range(W_1.shape[0]):
        for j in range(W_1.shape[1]):
            W_1_try = np.copy(W_1)
            W_1_try[i][j] -= h
            c1 = compute_cost(X, Y, W_1_try, W_2, b_1, b_2, lamda)[0]
            W_1_try = np.copy(W_1)
            W_1_try[i][j] += h
            c2 = compute_cost(X, Y, W_1_try, W_2, b_1, b_2, lamda)[0]
            grad_W_1_num[i][j] = (c2-c1) / (2*h)
    for i in range(W_2.shape[0]):
        for j in range(W_2.shape[1]):
            W_2_try = np.copy(W_2)
            W_2_try[i][j] -= h
            c1 = compute_cost(X, Y, W_1, W_2_try, b_1, b_2, lamda)[0]
            W_2_try = np.copy(W_2)
            W_2_try[i][j] += h
            c2 = compute_cost(X, Y, W_1, W_2_try, b_1, b_2, lamda)[0]
            grad_W_2_num[i][j] = (c2-c1) / (2*h)       


    return grad_b_1_num, grad_b_2_num, grad_W_1_num, grad_W_2_num

def check_grad(grad_b, grad_W, grad_b_num, grad_W_num, flag = 2):
    # Remember lamda = 0, so grad_b = J_grad_b and grad_W = J_grad_W 
    W_abs_diff = np.absolute(grad_W - grad_W_num)
    b_abs_diff = np.absolute(grad_b - grad_b_num)
    diff_W_average = np.average(W_abs_diff)
    if flag == 1:
        # When the gradient vectors have small values this approach may fail!
        if diff_W_average < 1e-5:
            print("Accepted! The absolute difference is less than 1e-6. (flag == 1)", '\n')
        else:
            print("average(W_abs_diff) =", diff_W_average)
            print("Warning...!", '\n')
    elif flag == 2:
        # More reliable method
        grad_W_abs = np.absolute(grad_W)
        grad_W_num_abs = np.absolute(grad_W_num)
        sum_W = grad_W_abs + grad_W_num_abs
        res = diff_W_average / np.amax(sum_W) 
        if res < 1e-5:
            print("res =", res)
            print("Accepted! The absolute difference is less than 1e-6. (flag == 2)", '\n')
        else:
            print("average(W_abs_diff) =", res)
            print("Warning...!", '\n')


# ----------------------------------------------------------------------------------------------------------- Mini-Batch --------->>>
def mini_batch(X, Y, X_val, Y_val_hot, GDparams, W_1, W_2, b_1, b_2, lamda, rho, drop_rate):     
    J_train_ls = []
    J_val_ls = []
    loss_train_ls = []
    loss_val_ls = []
    acc_train_ls = []
    acc_val_ls = []
    v_W1 = np.zeros((m, d))
    v_W2 = np.zeros((K, m))
    v_b1 = np.zeros((m, 1))
    v_b2 = np.zeros((K, 1))
    W_1_ls = []
    W_2_ls = []
    b_1_ls = [] 
    b_2_ls = []
    

    '''--- Initial stats ---''' 
    '''--- cost function and loss---'''
    J_train, loss_train = compute_cost(X, Y, W_1, W_2, b_1, b_2, lamda)
    J_val, loss_val = compute_cost(X_val, Y_val_hot, W_1, W_2, b_1, b_2, lamda)
    J_train_ls.append(J_train)
    J_val_ls.append(J_val)
    loss_train_ls.append(loss_train.sum() / X.shape[1])
    loss_val_ls.append(loss_val.sum() / X_val.shape[1])
    
    '''--- accuracy ---'''
    acc_train_ls.append(compute_accuracy(X, Y, W_1, W_2, b_1, b_2) * 100)
    acc_val_ls.append(compute_accuracy(X_val, Y_val_hot, W_1, W_2, b_1, b_2) * 100)

    
    
    # print("eta = " + str(GDparams.eta) + " | lamda = " + str(lamda))
    for epoch in range(GDparams.n_epochs):
        if J_train > 3 * 2.3:
            return "Warning!"
        print('Epoch: ' + str(epoch + 1))  
        i = 0
        while i < Y.shape[1]:
            if i + GDparams.n_batch > Y.shape[1]:
                X_batch = X[:, i:Y.shape[1]]
            else:
                X_batch = X[:, i:i + GDparams.n_batch]
            if i + GDparams.n_batch > Y.shape[1]:
                Y_batch = Y[:, i:Y.shape[1]]
            else:
                Y_batch = Y[:, i:i + GDparams.n_batch] 
           
            
            if drop_out == True:
                drop = np.random.choice([0, 1], size=(X_batch.shape[0],X_batch.shape[1]), p=[0.1, 0.9])
                X_b = np.multiply(X_batch, drop)
                P, h = dropout_evaluate_classifier(X_b, W_1, W_2, b_1, b_2)
            else:
                P, h = evaluate_classifier(X_batch, W_1, W_2, b_1, b_2)
            
            J_grad_W_1, J_grad_W_2, J_grad_b_1, J_grad_b_2 = compute_gradients(X_batch, Y_batch, P, h, W_1, W_2, b_1, b_2, lamda)

            
            # '''--- Check gradients: ---'''
            # print("Compute grads numerically ...")
            # grad_b_1_num, grad_b_2_num, grad_W_1_num, grad_W_2_num = compute_grads_num_slow(X_batch, Y_batch, W_1, W_2, b_1, b_2, lamda, h = 1e-5)
            # print('Start ckecking 1 ...')
            # check_grad(J_grad_b_1, J_grad_W_1, grad_b_1_num, grad_W_1_num)
            # print('Start ckecking 2 ...')
            # check_grad(J_grad_b_2, J_grad_W_2, grad_b_2_num, grad_W_2_num)
            # return None
            
            
            '''--- Momentum update ---'''
            v_W1 = rho * v_W1 + GDparams.eta * J_grad_W_1
            W_1 -=v_W1
            v_W2 = rho * v_W2 + GDparams.eta * J_grad_W_2
            W_2 -= v_W2
            v_b1 = rho * v_b1 + GDparams.eta * J_grad_b_1
            b_1 -= v_b1
            v_b2 = rho * v_b2 + GDparams.eta * J_grad_b_2
            b_2 -= v_b2
           
                
            i += GDparams.n_batch
                
        if drop_rate == True:
            GDparams.eta -= GDparams.eta - GDparams.eta * 0.95     
        
        J_train, loss_train = compute_cost(X, Y, W_1, W_2, b_1, b_2, lamda)
        J_val, loss_val = compute_cost(X_val, Y_val_hot, W_1, W_2, b_1, b_2, lamda)    
        # print("Epoch: " + str(epoch + 1) + " | training cost = " + str(J_train) + " | validation cost = " + str(J_val))
        
        '''--- cost function ---'''
        J_train_ls.append(J_train)
        J_val_ls.append(J_val)

        '''--- loss ---'''
        loss_train_ls.append(loss_train.sum() / X.shape[1])
        loss_val_ls.append(loss_val.sum() / X_val.shape[1])

        '''--- accuracy ---'''
        acc_train_ls.append(compute_accuracy(X, Y, W_1, W_2, b_1, b_2) * 100)
        acc_val_ls.append(compute_accuracy(X_val, Y_val_hot, W_1, W_2, b_1, b_2) * 100)
        max_train_acc = np.max(acc_train_ls)
        max_val_acc = np.max(acc_val_ls)
        
        W_1_ls.append(W_1)
        W_2_ls.append(W_2)
        b_1_ls.append(b_1)  
        b_2_ls.append(b_2)
        best_iter = np.argmax(acc_val_ls) 
    
    # print(" \n training accuracy = " + str(max_train_acc) + "%" + " | validation accuracy = " + str(max_val_acc) + "%", "\n")

    return J_train_ls, J_val_ls, loss_train_ls, loss_val_ls, acc_train_ls, acc_val_ls, W_1_ls[best_iter - 1], W_2_ls[best_iter - 1], b_1_ls[best_iter - 1], b_2_ls[best_iter - 1], max_train_acc, max_val_acc



def random_search(X, Y, X_val, Y_val_hot, GDparams, W_1, W_2, b_1, b_2, lamda, rho, drop_rate):
    best_res = [0, 0, 0, 0] 
    search_results = np.array([[0, 0, 0, 0]])
    '''--- update learning rate and lamda ---'''
    eta_rnd = 10 ** np.random.uniform(-6, 1, size = 10)
    lamda_rnd = 10 ** np.random.uniform(-6, 1, size = 10)

    for new_lamda in lamda_rnd:
        for new_eta in eta_rnd: 
            print("search_results = ", search_results)
            if best_res[-1] < search_results[-1, -1]:
                best_res = np.copy(search_results[-1, :])
                print("best_res = ", best_res)
            GDparams_updated = GD_params(n_batch = GDparams.n_batch, eta = new_eta, n_epochs = GDparams.n_epochs)
            max_train_acc, max_val_acc = mini_batch(X, Y, X_val, Y_val_hot, GDparams_updated, np.copy(W_1), np.copy(W_2), np.copy(b_1), np.copy(b_2), new_lamda, rho, drop_rate)[10:] 
            search_results = np.append(search_results, [[new_eta, new_lamda, max_train_acc, max_val_acc]], axis = 0)
    # np.savetxt('../../search_results.gz', search_results, fmt='%.5f', delimiter=' | ', newline='\n', header='', footer='', comments='# ')
    # np.savetxt('../../best_res.gz', best_res, fmt='%.5f', delimiter=' | ', newline='\n', header='', footer='', comments='# ')

    '''--- optimal area search ---'''
    best_res_around_opt = [0, 0, 0, 0]
    search_results_around_opt = np.array([[0, 0, 0, 0]])
    
    best_eta = np.copy(best_res[0])
    best_lamda = np.copy(best_res[1])
    eta_around_opt = np.random.normal(best_eta, best_eta / 2, size = 10)
    lamda_around_opt = np.random.normal(best_lamda, best_lamda / 2, size = 10)

    for lam in lamda_around_opt:
        for et in eta_around_opt: 
            print("search_results_around_opt = ", search_results_around_opt)
            if best_res_around_opt[-1] < search_results_around_opt[-1, -1]:
                best_res_around_opt = np.copy(search_results_around_opt[-1, :])
                print("best_res_around_opt = ", best_res_around_opt)
            GDparams_updated_new = GD_params(n_batch = GDparams.n_batch, eta = et, n_epochs = GDparams.n_epochs)
            max_train_acc, max_val_acc = mini_batch(X, Y, X_val, Y_val_hot, GDparams_updated_new, np.copy(W_1), np.copy(W_2), np.copy(b_1), np.copy(b_2), lam, rho, drop_rate)[10:] 
            search_results_around_opt = np.append(search_results_around_opt, [[et, lam, max_train_acc, max_val_acc]], axis = 0)
    # np.savetxt('../../search_results_around_opt.gz', search_results_around_opt, fmt='%.5f', delimiter=' | ', newline='\n', header='', footer='', comments='# ')
    # np.savetxt('../../best_res_around_opt.gz', best_res_around_opt, fmt='%.5f', delimiter=' | ', newline='\n', header='', footer='', comments='# ')


    return best_res_around_opt    
            

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
    plt.title('The maximum accuracy of the validation set is: ' + str(max_val_acc) + '%')
    plt.savefig("../figs/graphs" + ".png", bbox_inches='tight')
   

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
    # plt.savefig("../figs/weights_l=" + str(lamda) + "_b=" + str(GDparams.n_batch) + "_eta=" \
    #             + str(GDparams.eta) + "_ep=" + str(GDparams.n_epochs) + ".png", bbox_inches='tight')                        
   







def main():
    '''--- Load dataset ---'''
    print("Loading data ...")

    # X_train, Y_train_hot, y_train = load_data('../Datasets/cifar-10-batches-py/data_batch_1')
    # X_val, Y_val_hot, y_val = load_data('../Datasets/cifar-10-batches-py/data_batch_2')
    # X_test, Y_test_hot, y_test = load_data('../Datasets/cifar-10-batches-py/test_batch')
    # mean_X_tr = np.mean(X_train)
    # X_train -= mean_X_tr
    # X_val -= mean_X_tr
    # X_test -= mean_X_tr

    '''--- Use all the available data ---'''
    X_1, Y_1_hot, y_1 = load_data('../Datasets/cifar-10-batches-py/data_batch_1')
    X_2, Y_2_hot, y_2 = load_data('../Datasets/cifar-10-batches-py/data_batch_2')
    X_3, Y_3_hot, y_3 = load_data('../Datasets/cifar-10-batches-py/data_batch_3')
    X_4, Y_4_hot, y_4 = load_data('../Datasets/cifar-10-batches-py/data_batch_4')
    X_5, Y_5_hot, y_5 = load_data('../Datasets/cifar-10-batches-py/data_batch_5')
    X_all_train = np.concatenate((X_1, X_2, X_3, X_4, X_5), axis = 1)
    Y_all_train = np.concatenate((Y_1_hot, Y_2_hot, Y_3_hot, Y_4_hot, Y_5_hot), axis = 1)
    mean_X_all_train = np.mean(X_all_train)
    X_all_train -= mean_X_all_train

    '''--- training set ---'''
    X_train_new = X_all_train[:, :X_all_train.shape[1] - 1000]
    Y_train_new = Y_all_train[:, :Y_all_train.shape[1] - 1000]
    
    '''--- validation set ---'''
    '''--- Decrease the size of the validation set down to ~1000 --- '''
    X_val_new = X_all_train[:, X_all_train.shape[1] - 1000:]
    Y_val_hot_new = Y_all_train[:, Y_all_train.shape[1] - 1000:]
    
    '''--- test set ---'''
    X_test, Y_test_hot, y_test = load_data('../Datasets/cifar-10-batches-py/test_batch')
    X_test -= mean_X_all_train
    
    print("Done!") 


    '''--- Initialize model params ---'''
    mod_params = model_params()
    W_1 = np.copy(mod_params[0])
    W_2 = np.copy(mod_params[1])
    b_1 = np.copy(mod_params[2])
    b_2 = np.copy(mod_params[3])    


    GDparams = GD_params(n_batch = 100, eta = 0.01296, n_epochs = 50)

    '''--- Random search ---'''
    # best_res_around_opt = random_search(X_train, Y_train_hot, X_val, Y_val_hot, GDparams, W_1, W_2, b_1, b_2, lamda = 0.001, rho = .9, drop_rate = False)
    

    '''--- Mini-Batch ---'''
    
    '''------------------------------------------------------------------------------------ Part 1 ------------------------------------------------------------------------------------'''
    '''--- Only 1 data_batch ---'''
    # J_train_ls, J_val_ls, loss_train_ls, loss_val_ls, acc_train_ls, acc_val_ls, W_1_best, W_2_best, b_1_best, b_2_best, max_train_acc, max_val_acc = \
    #                                      mini_batch(X_train, Y_train_hot, X_val, Y_val_hot, GDparams, W_1, W_2, b_1, b_2, lamda = 0.00007, rho = .9, drop_rate = False)
    
    '''--- check for overfitting ---'''
    #J_train_ls, J_val_ls, loss_train_ls, loss_val_ls, acc_train_ls, acc_val_ls, W_1_best, W_2_best, b_1_best, b_2_best, max_train_acc, max_val_acc = \
                                        # mini_batch(X_train[:, :100], Y_train_hot[:, :100], X_val[:, :100], Y_val_hot[:, :100], GDparams, W_1, W_2, b_1, b_2, lamda = 0.00007, rho = .9, drop_rate = False)
    '''--- All available data ---'''
    J_train_ls, J_val_ls, loss_train_ls, loss_val_ls, acc_train_ls, acc_val_ls, W_1_best, W_2_best, b_1_best, b_2_best, max_train_acc, max_val_acc = \
                                        mini_batch(X_train_new, Y_train_new, X_val_new, Y_val_hot_new, GDparams, W_1, W_2, b_1, b_2, lamda = 0.00007, rho = .9, drop_rate = True)
    
    
    '''------------------------------------------------------------------------------------ Part 2 ------------------------------------------------------------------------------------'''
    '''--- (a) Train for a longer time and use your validation set to make sure you don’t overfit or keep a record of the best model before you begin to overfit. ---'''
    '''--- (c) You could also explore whether having more hidden nodes improves the final classification rate. One would expect that with more hidden nodes then the amount of regularization would have to increase. ---'''
    '''--- (e) Apply drop out to your training if you have a high number of hidden nodes and you feel you need more regularization. ---'''
    

    '''--- Test set classification ---'''
    cost, loss = compute_cost(X_test, Y_test_hot, W_1_best, W_2_best, b_1_best, b_2_best, lamda = 0.00007)
    test_acc = compute_accuracy(X_test, Y_test_hot, W_1_best, W_2_best, b_1_best, b_2_best) * 100
    max_test_acc = np.max(test_acc)
    print('test cost = ', cost)
    print('test loss = ', loss / X_test.shape[1])
    print('test accuracy = ', test_acc)
    

    '''--- Visualize ---'''
    graph_vis(J_train_ls, J_val_ls, loss_train_ls, loss_val_ls, acc_train_ls, acc_val_ls, GDparams, max_val_acc)
    #weights_vis(W_star, GDparams)
    plt.show()

if __name__ == "__main__":
    main()

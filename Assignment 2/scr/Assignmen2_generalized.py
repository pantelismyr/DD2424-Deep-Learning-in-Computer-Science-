import pickle as cPickle
import numpy as np
from timeit import default_timer
import matplotlib.pyplot as plt
import numpy.matlib
np.set_printoptions(formatter={'float_kind':lambda x: "%.5f" % x})

'''--- Global Var ---'''
d = 3072      # image dimention
K = 10        # total number of labels
N = 10000     # total number of images
m = [50, 30, K]        # nodes in the hidden layer
mu = 0        
sigma = .001
mode = 'ReLU'
drop_out = False
drop_rate = True

class GD_params:
    def __init__(self, n_batch, eta, n_epochs, lamda, rho, num_lay):
        self.n_batch = n_batch
        self.eta = eta
        self.n_epochs = n_epochs
        self.lamda = lamda
        self.rho = rho
        self.num_lay = num_lay                  # number of layers

    def model_params(self):
        # Initialize model parameters
        W = []
        b = []
        np.random.seed(400)
        W.append(np.random.normal(mu, sigma, (m[0], d)))
        b.append(np.zeros((m[0], 1)))
        for i in range(1, self.num_lay):    
            W.append(np.random.normal(mu, sigma, (m[i], m[i - 1])))
            b.append(np.zeros((m[i], 1)))   
        return W, b

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


def dropout_evaluate_classifier(X, W_1, W_2, b_1, b_2):
    s_1 = np.dot(W_1, X) + b_1
    h = activation_func(s_1)
    drop = np.random.choice([0, 1], size=(h.shape[0],h.shape[1]), p=[0.1, 0.9])
    h = np.multiply(h, drop)
    s = np.dot(W_2, h) + b_2
    softmax = np.exp(s) / np.sum(np.exp(s), axis = 0, keepdims = True)
    return softmax, h


def evaluate_classifier(X, W, b):
    h_ls = []
    
    for i in range(len(W)):
        h_ls.append(X)
        s = np.dot(W[i], X) + b[i]
        X = activation_func(s)
    
    
        
    softmax = np.exp(s) / np.sum(np.exp(s), axis = 0, keepdims = True)
    return softmax, h_ls

    
def activation_func(s):
    if mode == 'ReLU':
        h = np.maximum(0, s)
        return h
    elif mode == 'LeakyReLU':
        h = np.maximum(.01 * s, s)
        return h   

def compute_cost(X, Y, W, b, lamda):
    W_sum = 0
    for i in range(len(W)):
        W_ = W[i] ** 2
        W_sum += W_.sum()
    P, h = evaluate_classifier(X, W, b)
    l_cross = -np.log(np.diag(np.dot(Y.T, P)))              # when Y is encoded by a one-hot representation then l_cross=
    regularization = lamda * W_sum
    loss = l_cross.sum() 
    cost = loss / X.shape[1]  + regularization
    return cost, loss

def compute_accuracy(X, Y, W, b):
    P, h = evaluate_classifier(X, W, b)
    pred = np.argmax(P, axis = 0)
    eval_out = np.abs(pred - np.argmax(Y, axis = 0))
    error = np.count_nonzero(eval_out) / X.shape[1]
    return 1 - error    
# -------------------------------------------------------------------------------------------------------- Compute Gradients ----->>>
def compute_gradients(x, y, p, h, W, b, GDparams):
    '''
    compute gradients analytically 
    '''
    J_grad_W = []
    J_grad_b = []
    grad_W = {}
    grad_b = {}
    grad_W[0] = np.zeros((W[0].shape[0], W[0].shape[1]))
    grad_b[0] = np.zeros((b[0].shape[0], 1))

    g = p - y
    
    for i in reversed(range(GDparams.num_lay)):
        grad_W[i] = np.dot(g, h[i].T) / x.shape[1]
        grad_b[i] = g.sum() / x.shape[1]
        
   
        J_grad_W.append(grad_W[i] + 2 * GDparams.lamda * W[i])
        J_grad_b.append(grad_b[i])
        
        # Propagate the gradients
        g = np.dot(g.T, W[i])
        s_1 = np.copy(h[i])
        if mode == 'ReLU':
            ind = 1 * (s_1 > 0)
        elif mode == 'LeakyReLU':
            ind = (1 * (s_1 > 0) + 0.01 * (s_1 < 0))
        
        g = np.multiply(g.T, ind)
        
     
    return J_grad_W, J_grad_b
    
def compute_grads_num_slow(X, Y, GDparams, h = 1e-5):
    # compute gradients numerically
    W, b = GDparams.model_params()
    grad_W_num_ls = []
    for lay in range(GDparams.num_lay): 
        grad_W_num = np.zeros_like(W[lay])
        for i in range(W[lay].shape[0]):
            for j in range(W[lay].shape[1]):
                W_try = np.copy(W[lay])
                W_try[i][j] -= h
                W[lay] = W_try
                c1 = compute_cost(X, Y, W, b, GDparams.lamda)[0]
                W_try = np.copy(W[lay])
                W_try[i][j] += h
                W[lay] = W_try
                c2 = compute_cost(X, Y, W, b, GDparams.lamda)[0]
                grad_W_num[i][j] = (c2-c1) / (2*h)
        grad_W_num_ls.append(grad_W_num)
    return grad_W_num_ls

def check_grad(grad_W_num, J_grad_W, GDparams):
    for i in range(GDparams.num_lay):
        W_abs_diff = np.absolute(J_grad_W[i] - grad_W_num[i])
        diff_W_average = np.average(W_abs_diff)
        
        grad_W_abs = np.absolute(J_grad_W[i])
        grad_W_num_abs = np.absolute(grad_W_num[i])
        sum_W = grad_W_abs + grad_W_num_abs
        res = diff_W_average / np.amax(sum_W) 
        if res < 1e-5:
            print("res =", res)
            print("Accepted! The absolute difference is less than 1e-6. (flag == 2)", '\n')
        else:
            print("average(W_abs_diff) =", res)
            print("Warning...!", '\n')


# ----------------------------------------------------------------------------------------------------------- Mini-Batch --------->>>
def mini_batch(X, Y, X_val, Y_val_hot, GDparams, W, b):     
    J_train_ls = []
    J_val_ls = []
    loss_train_ls = []
    loss_val_ls = []
    acc_train_ls = []
    acc_val_ls = []
    
    W_store = []   #********
    b_store = []


    v_W = []
    v_b = []
    for i in range(GDparams.num_lay):
        v_W.append(np.zeros((W[i].shape[0], W[i].shape[1])))
        v_b.append(np.zeros((W[i].shape[0], 1)))
    

    '''--- Initial stats ---''' 
    '''--- cost function and loss---'''
    J_train, loss_train = compute_cost(X, Y, W, b, GDparams.lamda)
    J_val, loss_val = compute_cost(X_val, Y_val_hot, W, b, GDparams.lamda)
    J_train_ls.append(J_train)
    J_val_ls.append(J_val)
    loss_train_ls.append(loss_train.sum() / X.shape[1])
    loss_val_ls.append(loss_val.sum() / X_val.shape[1])
    
    '''--- accuracy ---'''
    acc_train_ls.append(compute_accuracy(X, Y, W, b) * 100)
    acc_val_ls.append(compute_accuracy(X_val, Y_val_hot, W, b) * 100)

    
    
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
                P, h_ls = evaluate_classifier(X_batch, W, b)
                
            
            J_grad_W, J_grad_b = compute_gradients(X_batch, Y_batch, P, np.copy(h_ls), W, b, GDparams)
            

            
            '''--- Check gradients: ---'''
            # print("Compute grads numerically ...")
            # grad_W_num = compute_grads_num_slow(X_batch, Y_batch, GDparams, h = 1e-5)
            # print("Checking grads ...")
            # grad_W_num.reverse()
            # check_grad(grad_W_num, J_grad_W, GDparams)
            # return None
            
            

            J_grad_W.reverse()      # sort the grads from 0 to num_lay ****
            
            '''--- Momentum update ---'''
            for j in range(GDparams.num_lay):
                v_W[j] = GDparams.rho * v_W[j] + GDparams.eta * J_grad_W[j]
                W[j] -= v_W[j]
                v_b[j] = GDparams.rho * v_b[j] + GDparams.eta * J_grad_b[j]
                b[j] -= v_b[j]
                
                
            i += GDparams.n_batch
                
        if drop_rate == True:
            GDparams.eta -= GDparams.eta - GDparams.eta * 0.95     
        
        J_train, loss_train = compute_cost(X, Y, W, b, GDparams.lamda)
        J_val, loss_val = compute_cost(X_val, Y_val_hot, W, b, GDparams.lamda)    
        # print("Epoch: " + str(epoch + 1) + " | training cost = " + str(J_train) + " | validation cost = " + str(J_val))
        
        '''--- cost function ---'''
        J_train_ls.append(J_train)
        J_val_ls.append(J_val)

        '''--- loss ---'''
        loss_train_ls.append(loss_train.sum() / X.shape[1])
        loss_val_ls.append(loss_val.sum() / X_val.shape[1])

        '''--- accuracy ---'''
        acc_train_ls.append(compute_accuracy(X, Y, W, b) * 100)
        acc_val_ls.append(compute_accuracy(X_val, Y_val_hot, W, b) * 100)
        max_train_acc = np.max(acc_train_ls)
        max_val_acc = np.max(acc_val_ls)

        W_store.append(W)
        b_store.append(b) 
        best_iter = np.argmax(acc_val_ls)
        
        
        W_best = W_store[best_iter - 1]
        b_best = b_store[best_iter - 1]
        
            
        
    
    # print(" \n training accuracy = " + str(max_train_acc) + "%" + " | validation accuracy = " + str(max_val_acc) + "%", "\n")

    return J_train_ls, J_val_ls, loss_train_ls, loss_val_ls, acc_train_ls, acc_val_ls, max_train_acc, max_val_acc, W_best, b_best



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
    # plt.savefig("../figs/graphs" + ".png", bbox_inches='tight')
   

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
 

    GDparams = GD_params(n_batch = 100, eta = .01296, n_epochs = 10, lamda = 0.00007, rho = .9, num_lay = 3)
    W, b = GDparams.model_params()
    
    '''--- Random search ---'''
    # best_res_around_opt = random_search(X_train, Y_train_hot, X_val, Y_val_hot, GDparams, W_1, W_2, b_1, b_2, lamda = 0.001, rho = .9, drop_rate = False)
    

    '''--- Mini-Batch ---'''
    J_train_ls, J_val_ls, loss_train_ls, loss_val_ls, acc_train_ls, acc_val_ls, max_train_acc, max_val_acc, W_best, b_best = \
                                                mini_batch(X_train_new[:,:10000], Y_train_new[:,:10000], X_val_new[:,:10000], Y_val_hot_new[:,:10000], GDparams, W, b)
   

    '''--- Test set classification ---'''
    # cost, loss = compute_cost(X_test, Y_test_hot, W_best, b_best, GDparams.lamda)
    # test_acc = compute_accuracy(X_test, Y_test_hot, W_best, b_best) * 100
    # max_test_acc = np.max(test_acc)
    # print('test cost = ', cost)
    # print('test loss = ', loss / X_test.shape[1])
    # print('test accuracy = ', test_acc)
    

    '''--- Visualize ---'''
    graph_vis(J_train_ls, J_val_ls, loss_train_ls, loss_val_ls, acc_train_ls, acc_val_ls, GDparams, max_val_acc)
    plt.show()

    


if __name__ == "__main__":
    main()

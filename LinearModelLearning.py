# -*- coding: utf-8 -*-
"""
Created on Thu Oct 01 15:06:10 2015

@author: shenoys
"""
import numpy as np
import matplotlib.pyplot as plt
#Linear regression experiments

def slope2D(x1, y1, x2, y2):
    return (y1 - y2) / (x1 - x2)

def target_func(x, y, m, c):
# y = f(x), where m is the slope of the decision boundary
    if m == 0:
        return np.sign(y);
    else:
        xi = 1.0 * (y - c) / m
        if x < xi:
            return -1
        else:
            return 1

def target_func_nonlinear(x):
    return np.sign(x[:, 1]**2 + x[:, 2]**2 - 0.6)
    
def hx(w, x):
    global N
    h = np.sign(x * w)
    return h

def init_trial_data(N):
    z = np.matrix(np.random.rand(2, 2) * 2 - 1)
    xn = np.matrix(np.random.rand(N, 2) * 2 - 1)
    xn = np.concatenate([np.matrix(np.ones([N, 1])), xn], axis=1)
    m = slope2D(z[0, 0], z[0, 1], z[1, 0], z[1, 1])
    yn = np.matrix(np.zeros([N, 1]))
    c = z[0, 1] - m * z[0, 0]
    #plt.plot(z[:, 0], z[:, 1], '-ko')
    for j in range(0, np.size(yn)):
        yn[j] = target_func(xn[j, 1], xn[j, 2], m, c)
#        if yn[j] == 1:
#            plt.plot(xn[j, 1], xn[j, 2], '-bx')
#        else:
#            plt.plot(xn[j, 1], xn[j, 2], '-rx')
    w = np.matrix(np.zeros([D + 1, 1]))
    return [xn, yn, w]

def init_trial_data_2(N):
    xn = np.matrix(np.random.rand(N, 2) * 2 - 1)
    xn = np.concatenate([np.matrix(np.ones([N, 1])), xn], axis=1)
    yn = np.matrix(np.zeros([N, 1]))

    for j in range(0, np.size(yn)):
        yn[j] = target_func_nonlinear(xn[j])
#        if yn[j] == 1:
#            plt.plot(xn[j, 1], xn[j, 2], '-bx')
#        else:
#            plt.plot(xn[j, 1], xn[j, 2], '-rx')
    #Add simulated noise to 10% of output samples
    for k in range(0, N/10):
        yn[10 * k + np.random.randint(0, 10)] = yn[10 * k + np.random.randint(0, 10)] * -1 
    w = np.matrix(np.zeros([D + 1, 1]))
    return [xn, yn, w]
    
def run_PLA(xn, yn, w, maxItr):
    if maxItr == 0:
        return 0
#    pointsDisagree = 0;
#    counter = 0;
    y_cap_idx = compute_error_vec(xn, yn, w)
    for n in range(0, maxItr):
        if np.size(y_cap_idx) > 0:
            w = w + np.transpose(xn[y_cap_idx[0, np.random.randint(0, np.size(y_cap_idx))]]) * yn[y_cap_idx[0, np.random.randint(0, np.size(y_cap_idx))]]
        else:
            #print("PLA has converged after %d interations" %n)
            break;
        y_cap_idx = compute_error_vec(xn, yn, w)

    if np.size(y_cap_idx) == 0 and n == maxItr:
        n += 1
#    if np.size(y_cap_idx) > 0:
#        pointsDisagree += np.size(y_cap_idx)
#        counter += 1
    return n

def run_linear_regression(xn, yn):
    global D
    #X_pseudo_inv =  np.linalg.inv(np.transpose(xn) * xn) * np.transpose(xn)
    X_pseudo_inv = np.linalg.pinv(xn)
    w = X_pseudo_inv * yn
    Ein = compute_Ein(compute_error_vec(xn, yn, w), np.size(yn))
    return [np.reshape(w, (1, D + 1)), Ein]

def compute_out_of_sample_error(xn, yn, g):
    Eout = compute_Ein(compute_error_vec(xn, yn, np.transpose(g)), np.size(yn))
    return Eout
    
def compute_error_vec(xn, yn, w):
    return np.where((hx(w, xn) != yn) == True)[0]
    
def compute_Ein(error_vec, size):
    return 1.0 * np.size(error_vec) / size  
    
if __name__ == "__main__":
    N = 1000   # Number of sample points
    D = 5
    M = 1000   #training data size
    totalTests = 1000
    maxItr = 1000    #max interations for PLA algorithm
    itrList = np.zeros([1, totalTests])
    Ein_list = np.zeros([1, totalTests])
    Eout_list = np.zeros([1, totalTests])
    g = np.matrix(np.zeros([totalTests, D + 1]))
    for i in range(0, totalTests):
#        [xn, yn, w] = init_trial_data(N + M)        
#        [g[i], Ein_list[0][i]] = run_linear_regression(xn[0:N], yn[0:N])
#        Eout_list[0][i] = compute_out_of_sample_error(xn[N:N + M], yn[N: N + M], g[i])
#        itrList[0][i] = run_PLA(xn[0:N], yn[0:N], np.transpose(g[i]), maxItr)
         [xn, yn, w] = init_trial_data_2(N + M)
         x1 = np.matrix(np.array(xn[:, 1]) * np.array(xn[:, 2]))
         x2 = np.matrix(np.array(xn[:, 1]) ** 2)
         x3 = np.matrix(np.array(xn[:, 2]) ** 2)
         xn = np.concatenate([xn, x1, x2, x3], axis=1)
         [g[i], Ein_list[0][i]] = run_linear_regression(xn[0:N], yn[0:N])
         Eout_list[0][i] = compute_out_of_sample_error(xn[N:N + M], yn[N: N + M], g[i])
    print("Average in sample error Ein %f" %np.average(Ein_list))
    print("Average out sample error Eout %f" %np.average(Eout_list))
    print("Average weight %s" %np.average(g, axis=0))

    print("Average number of iterations %d" %np.average(itrList[np.where(itrList != maxItr - 1)]))
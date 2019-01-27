#!/usr/bin/env python3



import numpy as np

import time


def least_squares_estimate_linear_regression_alg(in_X, in_Y):

    #print("in_X = \n", in_X)
    #print("in_Y = \n", in_Y)


    X_transp = in_X.T
    #print ("X_transp = \n",X_transp)

    XtX = np.dot(X_transp, in_X)
    #print("XtX = \n", XtX)
    
    XtX_inv = np.linalg.inv(XtX)
    #print ("XtX_inv = \n", XtX_inv)
    
    XtY = np.dot(X_transp, in_Y)
    #print("XtY = \n", XtY)
    
    out_W = np.dot(XtX_inv, XtY)
    #print("out_W = \n", out_W)   
    
    return out_W



def gradient_descent_linear_regression_alg(in_X, in_Y):

    #print("in_X = \n", in_X)
    #print("in_Y = \n", in_Y)



    X_transp = in_X.T
    #print ("X_transp = \n",X_transp)

    XtX = np.dot(X_transp, in_X)
    #print("XtX = \n", XtX)
    
    #XtX_inv = np.linalg.inv(XtX)
    #print ("XtX_inv = \n", XtX_inv)
    
    XtY = np.dot(X_transp, in_Y)
    #print("XtY = \n", XtY)
    
    
    
    max_iteration = 50000
    
    i = 1
    
    """
    
    
    beta_i = 0
    eta_0 = 0.1
    """
    
    epsilon = 0.0001
    
    #alpha_i = eta_0 / (1.0 + beta_i)
    
    alpha_i = 0.01 
    
    
    #init_W = np.ones((10, 1))
    
    
    tmp_X_shape = in_X.shape
    
    print("tmp_X_shape[0] = ", tmp_X_shape[0])
    print("tmp_X_shape[1] = ", tmp_X_shape[1])
    
    W_i = np.ones((tmp_X_shape[1], 1))
    #print("W_i = \n", W_i)
    
    while True:
            
        XtXW = np.dot(XtX, W_i)
        
        #print("XtXW = \n", XtXW)
    
        tmp_XtXW_minus_XtY = np.subtract(XtXW, XtY)
                
        #print("tmp_XtXW_minus_XtY = \n", tmp_XtXW_minus_XtY)
    
        W_i_plus_1 = np.subtract(W_i, 2 * alpha_i * tmp_XtXW_minus_XtY)
        
        #print("W_i_plus_1 = \n", W_i_plus_1)
        
        tmp_W_diff = np.subtract(W_i_plus_1, W_i)
        #print("tmp_W_diff = \n", tmp_W_diff)
        
        tmp_l2norm_of_W_diff = np.linalg.norm(tmp_W_diff, 2)
        #tmp_det_of_W_diff = np.linalg.det(tmp_W_diff)
        #print("tmp_l2norm_of_W_diff = \n", tmp_l2norm_of_W_diff)
    
        W_i = W_i_plus_1
        
        if abs(tmp_l2norm_of_W_diff) < epsilon:
            break
        
        i += 1
         
        
        if max_iteration < i:
            break
    
    
    #out_W = np.dot(XtX_inv, XtY)
    #print("out_W = \n", out_W)
    print("i =", i)
    
    return W_i




X = np.array([[0.86, 1], [0.09, 1], [-0.85, 1], [0.87, 1], [-0.44, 1], [-0.43, 1], [-1.10, 1], [0.40, 1], [-0.96, 1], [0.17, 1]])
#X = np.array([[0.75, 0.86, 1], [0.01, 0.09, 1], [0.73, -0.85, 1], [0.76, 0.87, 1], [0.19, -0.44, 1], [0.18, -0.43, 1], [1.22, -1.10, 1], [0.16, 0.40, 1], [0.93, -0.96, 1], [0.03, 0.17, 1]])

#X = np.array([[0.86], [0.09], [-0.85], [0.87], [-0.44], [-0.43], [-1.10], [0.40], [-0.96], [0.17]])



print("X = \n", X)
print(X.shape)


Y = np.array([[2.49], [0.83], [-0.25], [3.10], [0.87], [0.02], [-0.12], [1.81], [-0.83], [0.43]])
print("Y = \n", Y)
print(Y.shape)



start_time = time.time()
# your code

W = least_squares_estimate_linear_regression_alg(X, Y)

elapsed_time = time.time() - start_time
print("least_squares_estimate_linear_regression_alg W = \n", W)
print("elapsed_time = ", elapsed_time)


start_time = time.time()

W = gradient_descent_linear_regression_alg(X, Y)

elapsed_time = time.time() - start_time
print("gradient_descent_linear_regression_alg W = \n", W)
print("elapsed_time = ", elapsed_time)


#print("W = \n", W)

#W = [[1.61016842][1.05881341]]
#y = 1.60x + 1.05


#W = [[0.67437308] [1.74562466] [0.74315278]]
#y = 0.68x2 + 1.74x + 0.73





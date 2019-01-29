#!/usr/bin/env python3


import numpy as np

import math

import time


def mean_squared_error(in_A, in_B):
    
    diff_AB = np.subtract(in_A, in_B)
    
    print("diff_AB = \n", diff_AB)

    out_mse = np.square(diff_AB).mean()

    return out_mse

def least_squares_estimate_linear_regression_alg(in_X, in_Y):

    #print("in_X = \n", in_X)
    #print("in_Y = \n", in_Y)
    
    str_output = ""
    
    st = time.time()


    X_transp = in_X.T
    #print ("X_transp = \n",X_transp)
    
    et = time.time() - st
    
    str_tmp = "least_squares_estimate_linear_regression_alg elapsed time = " + str(et) + " after X_transp " + "\n"
    str_output += str_tmp
    
    print(str_tmp)           
    

    XtX = np.dot(X_transp, in_X)
    #print("XtX = \n", XtX)
    
    et = time.time() - st
    
    str_tmp = "least_squares_estimate_linear_regression_alg elapsed time = " + str(et) + " after XtX " + "\n"
    str_output += str_tmp
    
    print(str_tmp)
    
    
    XtX_inv = np.linalg.inv(XtX)
    #print ("XtX_inv = \n", XtX_inv)
    
    et = time.time() - st
    
    str_tmp = "least_squares_estimate_linear_regression_alg elapsed time = " + str(et) + " after XtX_inv " + "\n"
    str_output += str_tmp
    
    print(str_tmp)
    
    
    XtY = np.dot(X_transp, in_Y)
    
    et = time.time() - st
    
    str_tmp = "least_squares_estimate_linear_regression_alg elapsed time = " + str(et) + " after XtY " + "\n"
    str_output += str_tmp
    
    print(str_tmp)
    
    
    #print("XtY = \n", XtY)
    
    out_W = np.dot(XtX_inv, XtY)
    #print("out_W = \n", out_W) 
    
    et = time.time() - st
    
    str_tmp = "least_squares_estimate_linear_regression_alg elapsed time = " + str(et) + " after out_W " + "\n"
    str_output += str_tmp
    
    print(str_tmp)
    
    
    
    return out_W, str_output



#def gradient_descent_linear_regression_alg(in_X, in_Y, epsilon = 10**-6, beta_i = 10**-3, eta_0 = 10**-5):
    
def gradient_descent_linear_regression_alg(in_X, in_Y, epsilon = 10**-6, eta_0 = 1.0):

    #print("in_X = \n", in_X)
    #print("in_Y = \n", in_Y)
    
    str_output = ""


    X_transp = in_X.T
    #print ("X_transp = \n",X_transp)

    XtX = np.dot(X_transp, in_X)
    #print("XtX = \n", XtX)
    
    #XtX_inv = np.linalg.inv(XtX)
    #print ("XtX_inv = \n", XtX_inv)
    
    XtY = np.dot(X_transp, in_Y)
    #print("XtY = \n", XtY)
    
    
    
    max_iteration = 1000000
    
    i = 1
    beta_i = i
    k_i = 0
    
    
    """
    beta_i = 0
    eta_0 = 0.1
    """
    
    
    #epsilon = 0.0001
    
    #alpha_i = 0.01
    
    #checking interval
    k = 100
    
    
    tmp_X_shape = in_X.shape
    
    print("tmp_X_shape[0] = ", tmp_X_shape[0])
    print("tmp_X_shape[1] = ", tmp_X_shape[1])
    
    str_output = "tmp_X_shape[0] = " + str(tmp_X_shape[0]) + "\n"
    str_output += "tmp_X_shape[1] = " + str(tmp_X_shape[1]) + "\n"
    
    alpha_i = eta_0 / (1.0 + beta_i) / tmp_X_shape[0]   
         
    print("hyperparameters: eta_0, epsilon, beta_i, alpha_i : \n")
    
    str_output += "hyperparameters: eta_0, epsilon, beta_i, alpha_i : \n"
    
    print("eta_0 = ", eta_0)
    print("epsilon = ", epsilon)    
    print("beta_i = ", beta_i)   
    print("alpha_i = ", alpha_i)
    
    print("k = ", k)
   
    str_output += "eta_0 = " + str(eta_0) + "\n"
    str_output += "epsilon = " + str(epsilon) + "\n"
    str_output += "beta_i = " + str(beta_i) + "\n"
    str_output += "alpha_i = " + str(alpha_i) + "\n"
    str_output += "\n\n"
    
    str_output += "k = " + str(k) + "\n\n"
    
    
    #init_W = np.ones((10, 1))
    
    
    #W_i = np.ones((tmp_X_shape[1], 1))
    W_i = np.zeros((tmp_X_shape[1], 1))
    
    #print("W_i = \n", W_i)
    
    
    tmp_learning_rate_adjust_factor = 1.0
    
    tmp_l2norm_of_W_diff_last = 0.0
    
    st = time.time()
    
    while True:
            
        XtXW = np.dot(XtX, W_i)
        
        #print("XtXW = \n", XtXW)
    
        tmp_XtXW_minus_XtY = np.subtract(XtXW, XtY)
                
        #print("tmp_XtXW_minus_XtY = \n", tmp_XtXW_minus_XtY)
        
        #beta_i = i
        #k_i = i // 10
        #beta_i = math.pow(2, k_i)
        if i % 100 == 0:
            k_i = k_i + 1
            beta_i = math.pow(2, k_i)
        
        alpha_i = tmp_learning_rate_adjust_factor * eta_0 / (1.0 + beta_i) / tmp_X_shape[0]  
    
        
        tmp_double_alpha_XtXW_minus_XtY = 2 * alpha_i * tmp_XtXW_minus_XtY
    
        W_i_plus_1 = np.subtract(W_i, tmp_double_alpha_XtXW_minus_XtY)
        
        #print("W_i_plus_1 = \n", W_i_plus_1)
        
        #tmp_W_diff = np.subtract(W_i_plus_1, W_i)
        #tmp_double_alpha_XtXW_minus_XtY ~  - tmp_W_diff
        
        #print("tmp_W_diff = \n", tmp_W_diff)
        
        #tmp_l2norm_of_W_diff = np.linalg.norm(tmp_W_diff, 2)       
        tmp_l2norm_of_W_diff = np.linalg.norm(tmp_double_alpha_XtXW_minus_XtY, 2)
        
        #tmp_det_of_W_diff = np.linalg.det(tmp_W_diff)
        #print("tmp_l2norm_of_W_diff = ", tmp_l2norm_of_W_diff, " when i = ", i)        
        #str_output += "tmp_l2norm_of_W_diff = " + str(tmp_l2norm_of_W_diff) + " when i = " + str(i) + "\n"
    
        W_i = W_i_plus_1
        
        if abs(tmp_l2norm_of_W_diff) < epsilon:
            break
        
        if max_iteration < i:
            print("quit when exceed max_iteration: ", max_iteration)
            str_output += "quit when exceed max_iteration: " + str(max_iteration) + "\n"
            break
        
        if i % k == 0:
            #print("tmp_l2norm_of_W_diff = ", tmp_l2norm_of_W_diff)
            #print("tmp_l2norm_of_W_diff_last = ", tmp_l2norm_of_W_diff_last)
            
            #str_output += "tmp_l2norm_of_W_diff = " + str(tmp_l2norm_of_W_diff) + "\n"
            #str_output += "tmp_l2norm_of_W_diff_last = " + str(tmp_l2norm_of_W_diff_last) + "\n"            
            
            if i > k and abs(tmp_l2norm_of_W_diff) >= abs(tmp_l2norm_of_W_diff_last):
                
                tmp_learning_rate_adjust_factor *= 10**-1
                print("\n")
                print("set alpha_i to ", alpha_i, " when i = ", i, " for tmp_l2norm_of_W_diff = ", tmp_l2norm_of_W_diff, " tmp_l2norm_of_W_diff_last = ", tmp_l2norm_of_W_diff_last)
            
                str_output += "\n"
                str_output += "set alpha_i to " + str(alpha_i) + "\n"
                str_output += "when i = " + str(i) + "\n"
                str_output += "tmp_l2norm_of_W_diff = " + str(tmp_l2norm_of_W_diff) + "\n"
                str_output += "tmp_l2norm_of_W_diff_last = " + str(tmp_l2norm_of_W_diff_last) + "\n"
                
            
            
            tmp_l2norm_of_W_diff_last = tmp_l2norm_of_W_diff
        
        
        if i % 50000 == 0:
            et = time.time() - st
            print("elapsed time = ", et, " when iteration i = ", i)
            str_output += "elapsed time = " + str(et) + " when iteration i = " + str(i) + "\n"
                
        
        i += 1
        
        
        
    
    
    #out_W = np.dot(XtX_inv, XtY)
    #print("out_W = \n", out_W)
    print("\n\n total iterations i = ", i)
    print("\n\n")
    
    str_output += "\n\n total iterations i = " + str(i) + "\n\n"
    
    return W_i, str_output


"""

#X = np.array([[0.86, 1], [0.09, 1], [-0.85, 1], [0.87, 1], [-0.44, 1], [-0.43, 1], [-1.10, 1], [0.40, 1], [-0.96, 1], [0.17, 1]])
X = np.array([[0.75, 0.86, 1], [0.01, 0.09, 1], [0.73, -0.85, 1], [0.76, 0.87, 1], [0.19, -0.44, 1], [0.18, -0.43, 1], [1.22, -1.10, 1], [0.16, 0.40, 1], [0.93, -0.96, 1], [0.03, 0.17, 1]])

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
"""

#print("W = \n", W)

#W = [[1.61016842][1.05881341]]
#y = 1.60x + 1.05


#W = [[0.67437308] [1.74562466] [0.74315278]]
#y = 0.68x2 + 1.74x + 0.73

#MSE = np.square(np.subtract(A, B)).mean()





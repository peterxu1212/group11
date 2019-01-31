#!/usr/bin/env python3


import numpy as np

import math

import time


def mean_squared_error(in_A, in_B):
    
    diff_AB = np.subtract(in_A, in_B)
    
    #print("diff_AB = \n", diff_AB)

    Sigma_Square_of_diff_AB = np.square(diff_AB) 

    out_mse = Sigma_Square_of_diff_AB.mean()
    
    return out_mse, diff_AB, Sigma_Square_of_diff_AB

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
    
    try:
        
        XtX_inv = np.linalg.inv(XtX)
        #print ("XtX_inv = \n", XtX_inv)
    except Exception as e:
        
        str_tmp = "least_squares_estimate_linear_regression_alg Exception e = " + str(e) + " when execute np.linalg.inv for XtX = \n\n" + str(XtX) +  "\n\n"
        str_output += str_tmp
        
        print(str_tmp)
        return np.array([]), str_output
        
    
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
    
def gradient_descent_linear_regression_alg(in_X, in_Y, epsilon = 10**-6, eta_0 = 10**-1, distance_rate_power=4.0, T_huge_enough = 2000):

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
    
    tmp_count_after_Robbins_Monroe_stop = 0
    
    T_k = 10
    
    T_Robbins_Monroe = 100
    
    T_huge_enough = 5000
    
    
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
    print("distance_rate_power = ", distance_rate_power)
    print("T_huge_enough = ", T_huge_enough)
    
    print("T_k = ", T_k)
   
    str_output += "eta_0 = " + str(eta_0) + "\n"
    str_output += "epsilon = " + str(epsilon) + "\n"
    str_output += "beta_i = " + str(beta_i) + "\n"
    str_output += "alpha_i = " + str(alpha_i) + "\n"
    str_output += "distance_rate_power = " + str(distance_rate_power) + "\n"
    str_output += "T_huge_enough = " + str(T_huge_enough) + "\n"
    
    
    
    str_output += "\n\n"
    
    str_output += "T_k = " + str(T_k) + "\n\n"
    
    
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
        if i % T_Robbins_Monroe == 0:
            k_i = k_i + 1
            
            if (tmp_learning_rate_adjust_factor * eta_0 / (1.0 + beta_i)) >= (epsilon * 10.0**distance_rate_power):
                
                beta_i = math.pow(2, k_i)
            else:
                #beta_i should not change any more once alpha_i is small enough
                tmp_count_after_Robbins_Monroe_stop += T_Robbins_Monroe
                        
            print("\n")
            print("set k_i to ", k_i, " when i = ", i)
            print("set beta_i to ", beta_i, " when i = ", i)
        
            str_output += "\n"
            str_output += "set k_i to " + str(k_i) + "\n"
            str_output += "set beta_i to " + str(beta_i) + "\n"
            str_output += "when i = " + str(i) + "\n"
                
        
        alpha_i = tmp_learning_rate_adjust_factor * eta_0 / (1.0 + beta_i) / tmp_X_shape[0]
        
        
        if i % T_Robbins_Monroe == 0:
            print("set alpha_i to ", alpha_i, " when i = ", i)
            
            str_output += "\n"
            str_output += "set alpha_i to " + str(alpha_i) + "\n"

            str_output += "when i = " + str(i) + "\n"
    
        
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
            
            print("quit for tmp_l2norm_of_W_diff = ", tmp_l2norm_of_W_diff, " < epsilon when i = ", i)
            str_output += "quit for tmp_l2norm_of_W_diff = " + str(tmp_l2norm_of_W_diff) + " < epsilon when i = " + str(i) + "\n\n"            
            break
        
        if max_iteration < i:
            print("quit when exceed max_iteration: ", max_iteration)
            str_output += "quit when exceed max_iteration: " + str(max_iteration) + "\n"
            break
        
        if i % T_k == 0 and i % T_Robbins_Monroe != 0:
            #print("tmp_l2norm_of_W_diff = ", tmp_l2norm_of_W_diff)
            #print("tmp_l2norm_of_W_diff_last = ", tmp_l2norm_of_W_diff_last)
            
            #str_output += "tmp_l2norm_of_W_diff = " + str(tmp_l2norm_of_W_diff) + "\n"
            #str_output += "tmp_l2norm_of_W_diff_last = " + str(tmp_l2norm_of_W_diff_last) + "\n"            
            
            if i > T_k:
                if abs(tmp_l2norm_of_W_diff) >= abs(tmp_l2norm_of_W_diff_last):
                
                    tmp_learning_rate_adjust_factor *= 10**-1
                    print("\n")
                    print("set tmp_learning_rate_adjust_factor to ", tmp_learning_rate_adjust_factor, " when i = ", i, " for tmp_l2norm_of_W_diff = ", tmp_l2norm_of_W_diff, " tmp_l2norm_of_W_diff_last = ", tmp_l2norm_of_W_diff_last)
                
                    str_output += "\n"
                    str_output += "set tmp_learning_rate_adjust_factor to " + str(tmp_learning_rate_adjust_factor) + "\n"
                    str_output += "when i = " + str(i) + "\n"
                    str_output += "tmp_l2norm_of_W_diff = " + str(tmp_l2norm_of_W_diff) + "\n"
                    str_output += "tmp_l2norm_of_W_diff_last = " + str(tmp_l2norm_of_W_diff_last) + "\n"
                else:
                    if tmp_count_after_Robbins_Monroe_stop >= T_huge_enough:
                        tmp_count_after_Robbins_Monroe_stop = 0
                        
                        tmp_learning_rate_adjust_factor *= 0.5
                        print("\n")
                        print("set tmp_learning_rate_adjust_factor to ", tmp_learning_rate_adjust_factor, " when i = ", i, " for tmp_count_after_Robbins_Monroe_stop exceeds T_huge_enough = ", T_huge_enough)
                    
                        str_output += "\n"
                        str_output += "set tmp_learning_rate_adjust_factor to " + str(tmp_learning_rate_adjust_factor) + "\n"
                        str_output += "when i = " + str(i) + "\n"
                        str_output += "for tmp_count_after_Robbins_Monroe_stop exceeds T_huge_enough = " + str(T_huge_enough) + "\n"
                        
            
            tmp_l2norm_of_W_diff_last = tmp_l2norm_of_W_diff
        
        
        if i % T_huge_enough == 0:
            et = time.time() - st
            print("elapsed time = ", et, " when iteration i = ", i)
            str_output += "elapsed time = " + str(et) + " when iteration i = " + str(i) + "\n"
                
        
        i += 1
        
        
        
    
    
    #out_W = np.dot(XtX_inv, XtY)
    #print("out_W = \n", out_W)
    print("\n\n total iterations i = ", i)
    print("\n\n")
    
    str_output += "\n\n total iterations i = " + str(i) + "\n\n"
    
    str_output += "beta_i = " + str(beta_i) + "\n"
    str_output += "alpha_i = " + str(alpha_i) + "\n\n"
    
    str_output += "eta_0 = " + str(eta_0) + "\n"
    str_output += "epsilon = " + str(epsilon) + "\n"    
    str_output += "distance_rate_power = " + str(distance_rate_power) + "\n"
    str_output += "T_huge_enough = " + str(T_huge_enough) + "\n"
    
    
    str_output += "\n\n"
    
    str_output += "T_k = " + str(T_k) + "\n\n"
    
    return W_i, str_output, i



def gradient_descent_linear_regression_alg_origin(in_X, in_Y, epsilon = 10**-6, eta_0 = 10**-1, T_Robbins_Monroe=100):

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

            print("\n")
            print("set k_i to ", k_i, " when i = ", i)
            print("set beta_i to ", beta_i, " when i = ", i)

            str_output += "\n"
            str_output += "set k_i to " + str(k_i) + "\n"
            str_output += "set beta_i to " + str(beta_i) + "\n"
            str_output += "when i = " + str(i) + "\n"


        alpha_i = tmp_learning_rate_adjust_factor * eta_0 / (1.0 + beta_i) / tmp_X_shape[0]
        if i % 100 == 0:
            print("set alpha_i to ", alpha_i, " when i = ", i)

            str_output += "\n"
            str_output += "set alpha_i to " + str(alpha_i) + "\n"

            str_output += "when i = " + str(i) + "\n"


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
                print("set tmp_learning_rate_adjust_factor to ", tmp_learning_rate_adjust_factor, " when i = ", i, " for tmp_l2norm_of_W_diff = ", tmp_l2norm_of_W_diff, " tmp_l2norm_of_W_diff_last = ", tmp_l2norm_of_W_diff_last)

                str_output += "\n"
                str_output += "set tmp_learning_rate_adjust_factor to " + str(tmp_learning_rate_adjust_factor) + "\n"
                str_output += "when i = " + str(i) + "\n"
                str_output += "tmp_l2norm_of_W_diff = " + str(tmp_l2norm_of_W_diff) + "\n"
                str_output += "tmp_l2norm_of_W_diff_last = " + str(tmp_l2norm_of_W_diff_last) + "\n"



            tmp_l2norm_of_W_diff_last = tmp_l2norm_of_W_diff


        if i % 50000 == 0:
            et = time.time() - st
            print("elapsed time = ", et, " when iteration i = ", i)
            str_output += "elapsed time = " + str(et) + " when iteration i = " + str(i) + "\n"


        i += 1
        
    return W_i, str_output, i




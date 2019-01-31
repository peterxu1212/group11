# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 02:58:27 2019

@author: PeterXu
"""

import linear_regression as lr

import prepare_features as pf

import numpy as np

import time

    
    
import json # we need to use the JSON package to load the data, since the data is stored in JSON format


stat_data = []



#def 

str_for_statistics = ""

str_to_write = ""

with open('../t3_results.txt','w', buffering=1) as fout_t2r:

    #str_to_write = k + " " + str(v) + "\n"
    #fout_t2r.write(str_to_write)

    """
    #X = np.array([[0.86, 1], [0.09, 1], [-0.85, 1], [0.87, 1], [-0.44, 1], [-0.43, 1], [-1.10, 1], [0.40, 1], [-0.96, 1], [0.17, 1]])
    #X = np.array([[0.86], [0.09], [-0.85], [0.87], [-0.44], [-0.43], [-1.10], [0.40], [-0.96], [0.17]])
    
    X = np.array([[0.75, 0.86, 1], [0.01, 0.09, 1], [0.73, -0.85, 1], [0.76, 0.87, 1], [0.19, -0.44, 1], [0.18, -0.43, 1], [1.22, -1.10, 1], [0.16, 0.40, 1], [0.93, -0.96, 1], [0.03, 0.17, 1]])
    
    
    
    
    Y = np.array([[2.49], [0.83], [-0.25], [3.10], [0.87], [0.02], [-0.12], [1.81], [-0.83], [0.43]])
    #print("Y = \n", Y)
    #print(Y.shape)
    
    
    #print("X = \n", X)
    #print(X.shape)
    
    str_to_write = str(X) + "\n"
    fout_t2r.write(str_to_write)
    print(str_to_write)

    
    str_to_write = str(X.shape) + "\n"
    fout_t2r.write(str_to_write)
    print(str_to_write)
    
 
    str_to_write = str(Y) + "\n"
    fout_t2r.write(str_to_write)
    print(str_to_write)

    
    str_to_write = str(Y.shape) + "\n"
    fout_t2r.write(str_to_write)
    print(str_to_write)
    
    
    
    X_training_set, X_validation_set = np.split(X, [8])
    
    str_to_write = "X_training_set = \n" + str(X_training_set) + "\n"
    
    str_to_write += "X_validation_set = \n" + str(X_validation_set) + "\n"
    
    fout_t2r.write(str_to_write)
    print(str_to_write)
    
    #print("X_training_set = \n", X_training_set)
    #print("X_validation_set = \n", X_validation_set)
    
    Y_training_set, Y_validation_set = np.split(Y, [8])
    
    #print("Y_training_set = \n", Y_training_set)
    #print("Y_validation_set = \n", Y_validation_set)
    
    
    str_to_write = "Y_training_set = \n" + str(Y_training_set) + "\n"
    
    str_to_write += "Y_validation_set = \n" + str(Y_validation_set) + "\n"
    
    fout_t2r.write(str_to_write)
    print(str_to_write)
    """
    


    # retrieve word counts
    wordcount = {}
    
    wc_index = 0

    with open('../words.txt','r') as rf_words:
        
        while True:
            # read line
            line = rf_words.readline()
            #print(line)
            # check if line is not empty
            if not line:
                break
            
            lhs, rhs = line.split()        
            
            wordcount[lhs] = wc_index
            wc_index += 1
            #print(lhs, rhs)
            
        rf_words.close()
        
        
        
    X_training_set_Adv = np.array([])
    Y_training_set_Adv = np.array([])


    X_validation_set_Adv = np.array([])
    Y_validation_set_Adv = np.array([])
    
    
    running_stat_item = {}
    

    start_time = time.time()

    # generate feature for training set
    
    #with open("../testing_set.json", "r") as rf_training_set:   
    with open("../training_set.json", "r") as rf_training_set:
        data = json.load(rf_training_set)
        
      
    X_training_set_Adv, Y_training_set_Adv, str_output = pf.generate_wordfeature_and_output(wordcount, data, False, 0, False, 0)
    

    #X_training_set_Adv, Y_training_set_Adv = pf.generate_wordfeature_and_output(wordcount, data, True, 160, False, 0)
            
            
            
        #print(str_is_root, str_controversiality, str_children, str_popularity_score)
            
            
            
    str_to_write = "\n\n=================!!!!======================"  + "\n"
    
    str_to_write += "X_training_set_Adv = \n" + str(X_training_set_Adv) + "\n" + str(X_training_set_Adv.shape) + "\n"
    
    str_to_write += "Y_training_set_Adv = \n" + str(Y_training_set_Adv) + "\n" + str(Y_training_set_Adv.shape) + "\n"
    
    str_to_write += str_output
    
    fout_t2r.write(str_to_write)
    print(str_to_write)
    
    
    elapsed_time = time.time() - start_time
    
    
    #print("lr.least_squares_estimate_linear_regression_alg W = \n", W)
    #print("elapsed_time = ", elapsed_time)
    
    str_to_write = "prepare training set related matrix elapsed_time = " + str(elapsed_time) + "\n\n\n\n\n"
    
    fout_t2r.write(str_to_write)
    print(str_to_write)
    
    
    
    start_time = time.time()

    # generate feature for training set
    
    #with open("../testing_set.json", "r") as rf_training_set:   
    with open("../validation_set.json", "r") as rf_validation_set:
        data = json.load(rf_validation_set)
        
      
    X_validation_set_Adv, Y_validation_set_Adv, str_output = pf.generate_wordfeature_and_output(wordcount, data, False, 0, False, 0)
    

    #X_training_set_Adv, Y_training_set_Adv = pf.generate_wordfeature_and_output(wordcount, data, True, 160, False, 0)
            
            
            
        #print(str_is_root, str_controversiality, str_children, str_popularity_score)
            
            
            
    str_to_write = "\n\n=================!!!!======================"  + "\n"
    
    str_to_write += "X_validation_set_Adv = \n" + str(X_validation_set_Adv) + "\n" + str(X_validation_set_Adv.shape) + "\n"
    
    str_to_write += "Y_validation_set_Adv = \n" + str(Y_validation_set_Adv) + "\n" + str(Y_validation_set_Adv.shape) + "\n"
    
    str_to_write += str_output
    
    fout_t2r.write(str_to_write)
    print(str_to_write)
    
    
    elapsed_time = time.time() - start_time
    #print("lr.least_squares_estimate_linear_regression_alg W = \n", W)
    #print("elapsed_time = ", elapsed_time)
    
    str_to_write = "prepare validation set related matrix elapsed_time = " + str(elapsed_time) + "\n\n\n\n\n"
    
    fout_t2r.write(str_to_write)
    print(str_to_write)
    
    
    
    
    """ 
    
    start_time = time.time()
    # your code
    
    W, str_to_write = lr.least_squares_estimate_linear_regression_alg(X, Y)
    
    fout_t2r.write(str_to_write)
    print(str_to_write)
    
    elapsed_time = time.time() - start_time
    #print("lr.least_squares_estimate_linear_regression_alg W = \n", W)
    #print("elapsed_time = ", elapsed_time)
    
    str_to_write = "lr.least_squares_estimate_linear_regression_alg W = \n" + str(W) + "\n"
    str_to_write += "elapsed_time = " + str(elapsed_time) + "\n\n\n\n\n"
    
    fout_t2r.write(str_to_write)
    print(str_to_write)
    
    
    start_time = time.time()
    
    W, str_to_write = lr.gradient_descent_linear_regression_alg(X, Y)
    
    fout_t2r.write(str_to_write)
    print(str_to_write)
    
    elapsed_time = time.time() - start_time
    #print("lr.gradient_descent_linear_regression_alg W = \n", W)
    #print("elapsed_time = ", elapsed_time)
    
    str_to_write = "lr.gradient_descent_linear_regression_alg W = \n" + str(W) + "\n"
    str_to_write += "elapsed_time = " + str(elapsed_time) + "\n"
    
    fout_t2r.write(str_to_write)
    print(str_to_write)
    
    """
    
    
    
    
    
    
    
    
    
    
    str_to_write = "\n\n\n\n\n\n\n\n" + "for training and validation \n\n ===========!!!!===========\n\n"
    fout_t2r.write(str_to_write)
    print(str_to_write)
    
    
    start_time = time.time()
    
    W_cf, str_to_write = lr.least_squares_estimate_linear_regression_alg(X_training_set_Adv, Y_training_set_Adv)
    #print("lr.least_squares_estimate_linear_regression_alg W = \n", W)
    fout_t2r.write(str_to_write)
    print(str_to_write)
    
    
    str_to_write = "lr.least_squares_estimate_linear_regression_alg W_cf = \n" + str(W_cf) + "\n"
    
    elapsed_time = time.time() - start_time
    str_to_write += "least_squares_estimate_linear_regression_alg elapsed_time = " + str(elapsed_time) + "\n"
    
    fout_t2r.write(str_to_write)
    print(str_to_write)
    
    #print("elapsed_time = ", elapsed_time)
    
    str_for_statistics += "least_squares_estimate_linear_regression_alg:" + "\n\n"
    
    running_stat_item['desc'] = "least_squares_estimate_linear_regression_alg"
    
    str_for_statistics += "W_cf = \n" + str(W_cf) + "\n\n"
    
    running_stat_item['W'] = str(W_cf)
    
    str_for_statistics += "alg runtime = " + str(elapsed_time) + "\n"
    
    running_stat_item['alg_runtime'] = elapsed_time
    
    
    start_time = time.time()
    
    est_Y = np.dot(X_training_set_Adv, W_cf)
    tmp_mse, diff_AB, Sigma_Square_of_diff_AB = lr.mean_squared_error(est_Y, Y_training_set_Adv)
    
    #print("est_Y = \n", est_Y)
    #print("tmp_mse =", tmp_mse)
    
    elapsed_time = time.time() - start_time
    
    
    
    str_to_write = "tmp_mse for training set = " + str(tmp_mse) + "\n\n\n\n"
    str_to_write += "mean_squared_error elapsed_time = " + str(elapsed_time) + "\n"
        
    
    Sigma_Square_of_diff_AB = np.sort(Sigma_Square_of_diff_AB, axis=None, kind='mergesort')
    str_to_write += "Sigma_Square_of_diff_AB = \n" + str(Sigma_Square_of_diff_AB) + "\n" + str(Sigma_Square_of_diff_AB.shape) + "\n\n"

    
    str_to_write += "diff_AB = \n" + str(diff_AB) + "\n" + str(diff_AB.shape) + "\n\n\n\n"
    
    str_to_write += "est_Y = \n" + str(est_Y) + "\n" + str(est_Y.shape) + "\n"
    
    fout_t2r.write(str_to_write)
    print(str_to_write)
    
    
    str_for_statistics += "tmp_mse for training = " + str(tmp_mse) + "\n"
    
    running_stat_item['mse_for_training'] = tmp_mse
    
    
    
    start_time = time.time()
    
    est_Y = np.dot(X_validation_set_Adv, W_cf)
    tmp_mse, diff_AB, Sigma_Square_of_diff_AB = lr.mean_squared_error(est_Y, Y_validation_set_Adv)
    
    #print("est_Y = \n", est_Y)
    #print("tmp_mse =", tmp_mse)
    
    elapsed_time = time.time() - start_time
    
    
    
    str_to_write = "tmp_mse for validation set = " + str(tmp_mse) + "\n\n\n\n"
    str_to_write += "mean_squared_error elapsed_time = " + str(elapsed_time) + "\n"
        
    
    Sigma_Square_of_diff_AB = np.sort(Sigma_Square_of_diff_AB, axis=None, kind='mergesort')
    str_to_write += "Sigma_Square_of_diff_AB = \n" + str(Sigma_Square_of_diff_AB) + "\n" + str(Sigma_Square_of_diff_AB.shape) + "\n\n"

    
    str_to_write += "diff_AB = \n" + str(diff_AB) + "\n" + str(diff_AB.shape) + "\n\n\n\n"
    
    str_to_write += "est_Y = \n" + str(est_Y) + "\n" + str(est_Y.shape) + "\n"
    
    fout_t2r.write(str_to_write)
    print(str_to_write)
    
    
    
    str_for_statistics += "tmp_mse for validation = " + str(tmp_mse) + "\n"
    
    running_stat_item['mse_for_validation'] = tmp_mse
    
    
    
    start_time = time.time()
    
    tmp_mse, diff_AB, Sigma_Square_of_diff_AB = lr.mean_squared_error(W_cf, W_cf)
    
    #print("est_Y = \n", est_Y)
    #print("tmp_mse =", tmp_mse)
    
    elapsed_time = time.time() - start_time
    
    
    
    str_to_write = "tmp_mse for weight vector = " + str(tmp_mse) + "\n\n\n\n"
    str_to_write += "mean_squared_error elapsed_time = " + str(elapsed_time) + "\n"
        
    
    Sigma_Square_of_diff_AB = np.sort(Sigma_Square_of_diff_AB, axis=None, kind='mergesort')
    str_to_write += "Sigma_Square_of_diff_AB = \n" + str(Sigma_Square_of_diff_AB) + "\n" + str(Sigma_Square_of_diff_AB.shape) + "\n\n"

    
    str_to_write += "diff_AB = \n" + str(diff_AB) + "\n" + str(diff_AB.shape) + "\n\n\n\n"
    
    str_to_write += "W_cf = \n" + str(W_cf) + "\n" + str(W_cf.shape) + "\n"
    
    fout_t2r.write(str_to_write)
    print(str_to_write)
    
    
    
    str_for_statistics += "tmp_mse for weight vector = " + str(tmp_mse) + "\n"
    
        
    running_stat_item['mse_for_weight_vector'] = tmp_mse
    
    
    
    running_stat_item['total_execute_steps'] = 1
    
	
    running_stat_item['epsilon_power'] = 1
    running_stat_item['eta_power'] = 1
    running_stat_item['T_Robbins_Monroe_pow'] = 0

    
    running_stat_item['distance_rate_power'] = -1
    running_stat_item['T_huge_enough'] = -1
                    
    
    
    stat_data.append(running_stat_item)
    
    
    
    
    
    
    
    # report runtime, MSE and weights, and (stability), w.r.t differet hyperparameters
    
   
    # T_huge_enough
    
    # -3 ~ -8
    epsilon_power = -2
    
    # 0 ~ -3
    eta_power = 1
    
    # 2 ~ 5 
    distance_rate_power = 4
    
    #for T_huge_enough in (1000,):
    for T_huge_enough in (1000, 5000, 20000):
        
        for distance_rate_power in range(3, 8, 1):
           
            for eta_power in range(0, -4, -1): 
            #for eta_power in range(0, -4, -1):
            
                #for epsilon_power in range(-5, -10, -1):
                for epsilon_power in range(-4, -12, -1):
                
                    #print("\n\n\n\n")
                    
                    str_to_write = "\n\n\n\n"
            
                    fout_t2r.write(str_to_write)
                    print(str_to_write)
                    
                    
                    start_time = time.time()
                    
                    W, str_to_write, i_total_execute_steps = lr.gradient_descent_linear_regression_alg(X_training_set_Adv, Y_training_set_Adv, 10**epsilon_power, 10**eta_power, distance_rate_power)
                    
                    fout_t2r.write(str_to_write)
                    print(str_to_write)
                    
                    #print("lr.gradient_descent_linear_regression_alg W = \n", W)
                    
                    str_to_write = "lr.gradient_descent_linear_regression_alg W = \n" + str(W) + "\n"
            
                    fout_t2r.write(str_to_write)
                    print(str_to_write)
                    
                    elapsed_time = time.time() - start_time
                
                    #print("elapsed_time = ", elapsed_time)
                    
                    running_stat_item = {}
                    
                    str_for_statistics += "\n\n\n\n\n\n"
                    
                    str_for_statistics += "gradient_descent_linear_regression_alg:" + "\n\n"
                    
                    running_stat_item['desc'] = "gradient_descent_linear_regression_alg"
    
                    
                    str_for_statistics += "epsilon_power = " + str(epsilon_power) + "\n"                
                    str_for_statistics += "eta_power = " + str(eta_power) + "\n"                
                    str_for_statistics += "distance_rate_power = " + str(distance_rate_power) + "\n"                    
                    str_for_statistics += "T_huge_enough = " + str(T_huge_enough) + "\n"
                    
                    
                    
                    running_stat_item['epsilon_power'] = epsilon_power
                    running_stat_item['eta_power'] = eta_power
                                       
                    
                    running_stat_item['distance_rate_power'] = distance_rate_power
                    running_stat_item['T_huge_enough'] = T_huge_enough
                    
                    
                    running_stat_item['T_Robbins_Monroe_pow'] = 0

                    
                    
                    str_for_statistics += "W = \n" + str(W) + "\n\n"
                    
                    running_stat_item['W'] = str(W)
                    
                    str_for_statistics += "\n\n i_total_execute_steps = " + str(i_total_execute_steps) + "\n\n"
                           
                    running_stat_item['total_execute_steps'] = i_total_execute_steps
                    
                    str_for_statistics += "alg runtime = " + str(elapsed_time) + "\n"
                                       
                    running_stat_item['alg_runtime'] = elapsed_time
                    
                    est_Y = np.dot(X_training_set_Adv, W)
                    tmp_mse, diff_AB, Sigma_Square_of_diff_AB = lr.mean_squared_error(est_Y, Y_training_set_Adv)
                    
                    #print("est_Y = \n", est_Y)
                    #print("tmp_mse =", tmp_mse)
                    
                    str_to_write = "elapsed_time = " + str(elapsed_time) + "\n"
                    str_to_write += "tmp_mse for training set = " + str(tmp_mse) + "\n\n\n\n"
            
                    Sigma_Square_of_diff_AB = np.sort(Sigma_Square_of_diff_AB, axis=None, kind='mergesort')
                    str_to_write += "Sigma_Square_of_diff_AB = \n" + str(Sigma_Square_of_diff_AB) + "\n" + str(Sigma_Square_of_diff_AB.shape) + "\n\n"
            
                    
                    str_to_write += "diff_AB = \n" + str(diff_AB) + "\n" + str(diff_AB.shape) + "\n\n\n\n"
                    
                    str_to_write += "est_Y = \n" + str(est_Y) + "\n"
                    
                    
                    fout_t2r.write(str_to_write)
                    print(str_to_write)
                    
                    
                    
                    str_for_statistics += "tmp_mse for training = " + str(tmp_mse) + "\n"
                    
                    running_stat_item['mse_for_training'] = tmp_mse
                    
                    est_Y = np.dot(X_validation_set_Adv, W)
                    tmp_mse, diff_AB, Sigma_Square_of_diff_AB = lr.mean_squared_error(est_Y, Y_validation_set_Adv)
                    
                    #print("est_Y = \n", est_Y)
                    #print("tmp_mse =", tmp_mse)
                    
                    str_to_write = "elapsed_time = " + str(elapsed_time) + "\n"
                    str_to_write += "tmp_mse for validation set = " + str(tmp_mse) + "\n\n\n\n"
            
                    Sigma_Square_of_diff_AB = np.sort(Sigma_Square_of_diff_AB, axis=None, kind='mergesort')
                    str_to_write += "Sigma_Square_of_diff_AB = \n" + str(Sigma_Square_of_diff_AB) + "\n" + str(Sigma_Square_of_diff_AB.shape) + "\n\n"
            
                    
                    str_to_write += "diff_AB = \n" + str(diff_AB) + "\n" + str(diff_AB.shape) + "\n\n\n\n"
                    
                    str_to_write += "est_Y = \n" + str(est_Y) + "\n"
                    
                    
                    fout_t2r.write(str_to_write)
                    print(str_to_write)
                    
                    
                    
                    str_for_statistics += "tmp_mse for validation = " + str(tmp_mse) + "\n"
                    
                    running_stat_item['mse_for_validation'] = tmp_mse
                    
                    start_time = time.time()
        
                    tmp_mse, diff_AB, Sigma_Square_of_diff_AB = lr.mean_squared_error(W_cf, W)
                    
                    #print("est_Y = \n", est_Y)
                    #print("tmp_mse =", tmp_mse)
                    
                    elapsed_time = time.time() - start_time
                    
                    
                    
                    str_to_write = "tmp_mse for weight vector = " + str(tmp_mse) + "\n\n\n\n"
                    str_to_write += "mean_squared_error elapsed_time = " + str(elapsed_time) + "\n"
                        
                    
                    Sigma_Square_of_diff_AB = np.sort(Sigma_Square_of_diff_AB, axis=None, kind='mergesort')
                    str_to_write += "Sigma_Square_of_diff_AB = \n" + str(Sigma_Square_of_diff_AB) + "\n" + str(Sigma_Square_of_diff_AB.shape) + "\n\n"
                
                    
                    str_to_write += "diff_AB = \n" + str(diff_AB) + "\n" + str(diff_AB.shape) + "\n\n\n\n"
                    
                    str_to_write += "W = \n" + str(W) + "\n" + str(W.shape) + "\n"
                    
                    fout_t2r.write(str_to_write)
                    print(str_to_write)
                                        
                    
                    str_for_statistics += "tmp_mse for weight vector = " + str(tmp_mse) + "\n"
                
                    
                    running_stat_item['mse_for_weight_vector'] = tmp_mse
                
                    
                    stat_data.append(running_stat_item)
                
                
                  
    print(str_for_statistics)
    
    with open('../t3_result_stat.txt','w', buffering=1) as fout_t3rs:
        fout_t3rs.write(str_for_statistics)
    
    
    
    
    
    str_for_statistics = ""
    
    # -3 ~ -8
    epsilon_power = -2
    
    # 0 ~ -3
    eta_power = 1
    
    # 2 ~ 5 
    #distance_rate_power = 4
    
    for T_Robbins_Monroe_pow in range(1, 6, 1):
       
        for eta_power in range(0, -4, -1):    
        #for eta_power in range(0, -4, -1):
        
            #for epsilon_power in range(-5, -10, -1):
            for epsilon_power in range(-3, -12, -1):
            
                #print("\n\n\n\n")
                
                str_to_write = "\n\n\n\n"
        
                fout_t2r.write(str_to_write)
                print(str_to_write)
                
                
                start_time = time.time()
                
                #W, str_to_write = lr.gradient_descent_linear_regression_alg_old(X_training_set_Adv, Y_training_set_Adv, 10**epsilon_power, 10**eta_power, distance_rate_power)
                
                W, str_to_write, i_total_execute_steps = lr.gradient_descent_linear_regression_alg_old(X_training_set_Adv, Y_training_set_Adv, 10**epsilon_power, 10**eta_power, 10**T_Robbins_Monroe_pow)
                
                fout_t2r.write(str_to_write)
                print(str_to_write)
                
                #print("lr.gradient_descent_linear_regression_alg W = \n", W)
                
                str_to_write = "lr.gradient_descent_linear_regression_alg_old W = \n" + str(W) + "\n"
        
                fout_t2r.write(str_to_write)
                print(str_to_write)
                
                elapsed_time = time.time() - start_time
            
                #print("elapsed_time = ", elapsed_time)
                
                running_stat_item = {}
                
                
                #stat_data.append(running_stat_item)
                
                str_for_statistics += "\n\n\n\n\n\n"
                
                str_for_statistics += "gradient_descent_linear_regression_alg_old:" + "\n\n"
                
                running_stat_item['desc'] = "gradient_descent_linear_regression_alg_old"
                
                str_for_statistics += "epsilon_power = " + str(epsilon_power) + "\n"                
                str_for_statistics += "eta_power = " + str(eta_power) + "\n"                
                str_for_statistics += "T_Robbins_Monroe_pow = " + str(T_Robbins_Monroe_pow) + "\n"
                
                running_stat_item['epsilon_power'] = epsilon_power
                running_stat_item['eta_power'] = eta_power
                running_stat_item['T_Robbins_Monroe_pow'] = T_Robbins_Monroe_pow
                
                
                running_stat_item['distance_rate_power'] = -1
                running_stat_item['T_huge_enough'] = -1
                
                
                str_for_statistics += "W = \n" + str(W) + "\n\n"
                
                running_stat_item['W'] = str(W)
                
                str_for_statistics += "\n\n i_total_execute_steps = " + str(i_total_execute_steps) + "\n\n"
                
                running_stat_item['total_execute_steps'] = i_total_execute_steps
    
                str_for_statistics += "alg runtime = " + str(elapsed_time) + "\n"
                                
                running_stat_item['alg_runtime'] = elapsed_time
                
                
                est_Y = np.dot(X_training_set_Adv, W)
                tmp_mse, diff_AB, Sigma_Square_of_diff_AB = lr.mean_squared_error(est_Y, Y_training_set_Adv)
                
                #print("est_Y = \n", est_Y)
                #print("tmp_mse =", tmp_mse)
                
                str_to_write = "elapsed_time = " + str(elapsed_time) + "\n"
                str_to_write += "tmp_mse for training set = " + str(tmp_mse) + "\n\n\n\n"
        
                Sigma_Square_of_diff_AB = np.sort(Sigma_Square_of_diff_AB, axis=None, kind='mergesort')
                str_to_write += "Sigma_Square_of_diff_AB = \n" + str(Sigma_Square_of_diff_AB) + "\n" + str(Sigma_Square_of_diff_AB.shape) + "\n\n"
        
                
                str_to_write += "diff_AB = \n" + str(diff_AB) + "\n" + str(diff_AB.shape) + "\n\n\n\n"
                
                str_to_write += "est_Y = \n" + str(est_Y) + "\n"
                
                
                fout_t2r.write(str_to_write)
                print(str_to_write)
                
                
                
                str_for_statistics += "tmp_mse for training = " + str(tmp_mse) + "\n"
                
                
                running_stat_item['mse_for_training'] = tmp_mse
                
                est_Y = np.dot(X_validation_set_Adv, W)
                tmp_mse, diff_AB, Sigma_Square_of_diff_AB = lr.mean_squared_error(est_Y, Y_validation_set_Adv)
                
                #print("est_Y = \n", est_Y)
                #print("tmp_mse =", tmp_mse)
                
                str_to_write = "elapsed_time = " + str(elapsed_time) + "\n"
                str_to_write += "tmp_mse for validation set = " + str(tmp_mse) + "\n\n\n\n"
        
                Sigma_Square_of_diff_AB = np.sort(Sigma_Square_of_diff_AB, axis=None, kind='mergesort')
                str_to_write += "Sigma_Square_of_diff_AB = \n" + str(Sigma_Square_of_diff_AB) + "\n" + str(Sigma_Square_of_diff_AB.shape) + "\n\n"
        
                
                str_to_write += "diff_AB = \n" + str(diff_AB) + "\n" + str(diff_AB.shape) + "\n\n\n\n"
                
                str_to_write += "est_Y = \n" + str(est_Y) + "\n"
                
                
                fout_t2r.write(str_to_write)
                print(str_to_write)
                
                
                
                str_for_statistics += "tmp_mse for validation = " + str(tmp_mse) + "\n"
                
                
                running_stat_item['mse_for_validation'] = tmp_mse
                
                
                
                start_time = time.time()
    
                tmp_mse, diff_AB, Sigma_Square_of_diff_AB = lr.mean_squared_error(W_cf, W)
                
                #print("est_Y = \n", est_Y)
                #print("tmp_mse =", tmp_mse)
                
                elapsed_time = time.time() - start_time
                
                
                
                str_to_write = "tmp_mse for weight vector = " + str(tmp_mse) + "\n\n\n\n"
                str_to_write += "mean_squared_error elapsed_time = " + str(elapsed_time) + "\n"
                    
                
                Sigma_Square_of_diff_AB = np.sort(Sigma_Square_of_diff_AB, axis=None, kind='mergesort')
                str_to_write += "Sigma_Square_of_diff_AB = \n" + str(Sigma_Square_of_diff_AB) + "\n" + str(Sigma_Square_of_diff_AB.shape) + "\n\n"
            
                
                str_to_write += "diff_AB = \n" + str(diff_AB) + "\n" + str(diff_AB.shape) + "\n\n\n\n"
                
                str_to_write += "W = \n" + str(W) + "\n" + str(W.shape) + "\n"
                
                fout_t2r.write(str_to_write)
                print(str_to_write)
                
                
                
                str_for_statistics += "tmp_mse for weight vector = " + str(tmp_mse) + "\n"
                
                
                running_stat_item['mse_for_weight_vector'] = tmp_mse
                
                
                stat_data.append(running_stat_item)
                
                
                  
    print(str_for_statistics)
    
    with open('../t3_result_stat_old.txt','w', buffering=1) as fout_t3rs:
        fout_t3rs.write(str_for_statistics)
        
    
    with open("../task3_stat.json", "w") as wf_json_set:
        json.dump(stat_data, wf_json_set)
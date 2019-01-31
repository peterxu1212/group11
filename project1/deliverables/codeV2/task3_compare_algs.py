# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 02:58:27 2019

"""

import linear_regression as lr

import prepare_features as pf

import numpy as np

import time

    
    
import json # we need to use the JSON package to load the data, since the data is stored in JSON format


stat_data = []



 

str_for_statistics = ""

str_to_write = ""

with open('../t3_results.txt','w', buffering=1) as fout_t2r:



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

    
    X_testing_set_Adv = np.array([])
    Y_testing_set_Adv = np.array([])    
    
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
    
    
    
    
    
    
    
    
    
    start_time = time.time()

    # generate feature for training set
    
    #with open("../testing_set.json", "r") as rf_training_set:   
    with open("../testing_set.json", "r") as rf_testing_set:
        data = json.load(rf_testing_set)
        
      
    X_testing_set_Adv, Y_testing_set_Adv, str_output = pf.generate_wordfeature_and_output(wordcount, data, False, 0, False, 0)
            
            
            
    str_to_write = "\n\n=================!!!!======================"  + "\n"
    
    str_to_write += "X_testing_set_Adv = \n" + str(X_testing_set_Adv) + "\n" + str(X_testing_set_Adv.shape) + "\n"
    
    str_to_write += "Y_testing_set_Adv = \n" + str(Y_testing_set_Adv) + "\n" + str(Y_testing_set_Adv.shape) + "\n"
    
    str_to_write += str_output
    
    fout_t2r.write(str_to_write)
    print(str_to_write)
    
    
    elapsed_time = time.time() - start_time
    #print("lr.least_squares_estimate_linear_regression_alg W = \n", W)
    #print("elapsed_time = ", elapsed_time)
    
    str_to_write = "prepare testing set related matrix elapsed_time = " + str(elapsed_time) + "\n\n\n\n\n"
    
    fout_t2r.write(str_to_write)
    print(str_to_write)
    
    
    
    
    
    
    
    
    
    
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
    
    est_Y = np.dot(X_testing_set_Adv, W_cf)
    tmp_mse, diff_AB, Sigma_Square_of_diff_AB = lr.mean_squared_error(est_Y, Y_testing_set_Adv)
    
    #print("est_Y = \n", est_Y)
    #print("tmp_mse =", tmp_mse)
    
    elapsed_time = time.time() - start_time
    
    
    
    str_to_write = "tmp_mse for testing set = " + str(tmp_mse) + "\n\n\n\n"
    str_to_write += "mean_squared_error elapsed_time = " + str(elapsed_time) + "\n"
        
    
    Sigma_Square_of_diff_AB = np.sort(Sigma_Square_of_diff_AB, axis=None, kind='mergesort')
    str_to_write += "Sigma_Square_of_diff_AB = \n" + str(Sigma_Square_of_diff_AB) + "\n" + str(Sigma_Square_of_diff_AB.shape) + "\n\n"

    
    str_to_write += "diff_AB = \n" + str(diff_AB) + "\n" + str(diff_AB.shape) + "\n\n\n\n"
    
    str_to_write += "est_Y = \n" + str(est_Y) + "\n" + str(est_Y.shape) + "\n"
    
    fout_t2r.write(str_to_write)
    print(str_to_write)
    
    
    
    str_for_statistics += "tmp_mse for testing = " + str(tmp_mse) + "\n"
    
    running_stat_item['mse_for_testing'] = tmp_mse
    
    
    
    
    
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
    for T_huge_enough in (1000, 5000):
        
        for distance_rate_power in range(3, 9, 1):
           
            for eta_power in range(0, -4, -1): 
            #for eta_power in range(0, -4, -1):
            
                #for epsilon_power in range(-5, -10, -1):
                for epsilon_power in range(-4, -10, -1):
                
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
                    
                    
                    
                    est_Y = np.dot(X_testing_set_Adv, W)
                    tmp_mse, diff_AB, Sigma_Square_of_diff_AB = lr.mean_squared_error(est_Y, Y_testing_set_Adv)
                    
                    #print("est_Y = \n", est_Y)
                    #print("tmp_mse =", tmp_mse)
                    
                    str_to_write = "elapsed_time = " + str(elapsed_time) + "\n"
                    str_to_write += "tmp_mse for testing set = " + str(tmp_mse) + "\n\n\n\n"
            
                    Sigma_Square_of_diff_AB = np.sort(Sigma_Square_of_diff_AB, axis=None, kind='mergesort')
                    str_to_write += "Sigma_Square_of_diff_AB = \n" + str(Sigma_Square_of_diff_AB) + "\n" + str(Sigma_Square_of_diff_AB.shape) + "\n\n"
            
                    
                    str_to_write += "diff_AB = \n" + str(diff_AB) + "\n" + str(diff_AB.shape) + "\n\n\n\n"
                    
                    str_to_write += "est_Y = \n" + str(est_Y) + "\n"
                    
                    
                    fout_t2r.write(str_to_write)
                    print(str_to_write)
                    
                    
                    
                    str_for_statistics += "tmp_mse for testing = " + str(tmp_mse) + "\n"
                    
                    running_stat_item['mse_for_testing'] = tmp_mse
                    
                    
                    
                    
                    
                    
                    
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
                
               
                W, str_to_write, i_total_execute_steps = lr.gradient_descent_linear_regression_alg_origin(X_training_set_Adv, Y_training_set_Adv, 10**epsilon_power, 10**eta_power, 10**T_Robbins_Monroe_pow)
                
                fout_t2r.write(str_to_write)
                print(str_to_write)
                
                #print("lr.gradient_descent_linear_regression_alg W = \n", W)
                
                str_to_write = "lr.gradient_descent_linear_regression_alg_origin W = \n" + str(W) + "\n"
        
                fout_t2r.write(str_to_write)
                print(str_to_write)
                
                elapsed_time = time.time() - start_time
            
                #print("elapsed_time = ", elapsed_time)
                
                running_stat_item = {}
                
                
                #stat_data.append(running_stat_item)
                
                str_for_statistics += "\n\n\n\n\n\n"
                
                str_for_statistics += "gradient_descent_linear_regression_alg_origin:" + "\n\n"
                
                running_stat_item['desc'] = "gradient_descent_linear_regression_alg_origin"
                
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
                
                
                
                
                
                
                
                est_Y = np.dot(X_testing_set_Adv, W)
                tmp_mse, diff_AB, Sigma_Square_of_diff_AB = lr.mean_squared_error(est_Y, Y_testing_set_Adv)
                
                #print("est_Y = \n", est_Y)
                #print("tmp_mse =", tmp_mse)
                
                str_to_write = "elapsed_time = " + str(elapsed_time) + "\n"
                str_to_write += "tmp_mse for testing set = " + str(tmp_mse) + "\n\n\n\n"
        
                Sigma_Square_of_diff_AB = np.sort(Sigma_Square_of_diff_AB, axis=None, kind='mergesort')
                str_to_write += "Sigma_Square_of_diff_AB = \n" + str(Sigma_Square_of_diff_AB) + "\n" + str(Sigma_Square_of_diff_AB.shape) + "\n\n"
        
                
                str_to_write += "diff_AB = \n" + str(diff_AB) + "\n" + str(diff_AB.shape) + "\n\n\n\n"
                
                str_to_write += "est_Y = \n" + str(est_Y) + "\n"
                
                
                fout_t2r.write(str_to_write)
                print(str_to_write)
                
                
                
                str_for_statistics += "tmp_mse for testing = " + str(tmp_mse) + "\n"
                
                
                running_stat_item['mse_for_testing'] = tmp_mse
                
                
                
                
                
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
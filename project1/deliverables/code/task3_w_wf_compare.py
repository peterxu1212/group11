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


str_for_statistics = ""

stat_data = []


str_to_write = ""

with open('../t3_results_w_wf_compare.txt','w', buffering=1) as fout_t2r:

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















    for word_file_name in ("words", "words_60", "words_160", "words_260", "words_300", "words_60_adv", "words_160_adv", "words_260_adv", "words_300_adv"):

        i_len_word_feature = 0
        b_with_wf = False
        
        #include also sw
        b_without_punctuation = False
        b_without_stopwords = False
        
        if word_file_name != "words":
            
            b_with_wf = True
                        
            wlist = word_file_name.split('_')
            
            i_len_word_feature = int(wlist[1])
                      
            if len(wlist) >=3:
                b_without_punctuation = True
                b_without_stopwords = True
                
        
        
        
        
        
        # retrieve word counts
        wordcount = {}
        
        wc_index = 0
    
        str_full_file_name = '../' + word_file_name + '.txt'
        with open(str_full_file_name,'r') as rf_words:
            
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
            data_training_set = json.load(rf_training_set)
            
                   
        X_training_set_Adv, Y_training_set_Adv, str_output = pf.generate_wordfeature_and_output(wordcount, data_training_set, b_with_wf, i_len_word_feature, False, 0, b_without_punctuation, False, False, False, False, b_without_stopwords)
                
        #print(str_is_root, str_controversiality, str_children, str_popularity_score)
                
                
                
        str_to_write = "\n\n=================!!!!======================"  + "\n"
        
        str_to_write += "X_training_set_Adv = \n" + str(X_training_set_Adv) + "\n" + str(X_training_set_Adv.shape) + "\n"
        
        str_to_write += "Y_training_set_Adv = \n" + str(Y_training_set_Adv) + "\n" + str(Y_training_set_Adv.shape) + "\n"
        
        str_to_write += str_output
        
        fout_t2r.write(str_to_write)
        print(str_to_write)
        
        
        elapsed_time = time.time() - start_time
        
        running_stat_item['pre_alg_runtime'] = elapsed_time
        
        #print("lr.least_squares_estimate_linear_regression_alg W = \n", W)
        #print("elapsed_time = ", elapsed_time)
        
        str_to_write = "prepare training set related matrix elapsed_time = " + str(elapsed_time) + "\n\n\n\n\n"
        
        fout_t2r.write(str_to_write)
        print(str_to_write)
        
        
        
        start_time = time.time()
    
        # generate feature for training set
        
        #with open("../testing_set.json", "r") as rf_training_set:   
        with open("../validation_set.json", "r") as rf_validation_set:
            data_validation_set = json.load(rf_validation_set)
            
          
        #X_validation_set_Adv, Y_validation_set_Adv = pf.generate_wordfeature_and_output(wordcount, data, False, 0, False, 0)
        X_validation_set_Adv, Y_validation_set_Adv, str_output = pf.generate_wordfeature_and_output(wordcount, data_validation_set, b_with_wf, i_len_word_feature, False, 0, b_without_punctuation, False, False, False, False, b_without_stopwords)
                
        #print(str_is_root, str_controversiality, str_children, str_popularity_score)
                
                
                
        str_to_write = "\n\n=================!!!!======================"  + "\n"
        
        str_to_write += "X_validation_set_Adv = \n" + str(X_validation_set_Adv) + "\n" + str(X_validation_set_Adv.shape) + "\n"
        
        str_to_write += "Y_validation_set_Adv = \n" + str(Y_validation_set_Adv) + "\n" + str(Y_validation_set_Adv.shape) + "\n"
        
        
        str_to_write += str_output
        
        fout_t2r.write(str_to_write)
        print(str_to_write)
        
        
        elapsed_time = time.time() - start_time
                
        running_stat_item['pre_alg_runtime'] += elapsed_time
        
        #print("lr.least_squares_estimate_linear_regression_alg W = \n", W)
        #print("elapsed_time = ", elapsed_time)
        
        str_to_write = "prepare validation set related matrix elapsed_time = " + str(elapsed_time) + "\n\n\n\n\n"
        
        fout_t2r.write(str_to_write)
        print(str_to_write)
        
                
        
        
        
        str_tmp = "for word_file_name = " + word_file_name + "\n\n"
        
        str_tmp += "i_len_word_feature = " + str(i_len_word_feature) + "\n"                
        str_tmp += "b_with_wf = " + str(b_with_wf) + "\n"                
        str_tmp += "b_without_punctuation = " + str(b_without_punctuation) + "\n"
        str_tmp += "b_without_stopwords = " + str(b_without_stopwords) + "\n"
        
        
        
        print(str_tmp)
        fout_t2r.write(str_tmp)
        
        
        str_to_write = "\n\n\n\n\n\n\n\n" + "for training and validation \n\n ===========!!!!===========\n\n"
        fout_t2r.write(str_to_write)
        print(str_to_write)
        
        
        
        start_time = time.time()
        
        W, str_to_write = lr.least_squares_estimate_linear_regression_alg(X_training_set_Adv, Y_training_set_Adv)
        #print("lr.least_squares_estimate_linear_regression_alg W = \n", W)
        fout_t2r.write(str_to_write)
        print(str_to_write)
        
        elapsed_time = time.time() - start_time
        
        
        
        running_stat_item['desc'] = "least_squares_estimate_linear_regression_alg"
        
        
        running_stat_item['failure'] = 0
        
            
        running_stat_item['i_word_count_feature_count'] = i_len_word_feature            
        running_stat_item['b_with_word_count_feature'] = b_with_wf
        running_stat_item['b_without_punctuation'] = b_without_punctuation
        running_stat_item['b_without_stopwords'] = b_without_stopwords  
        
        
        running_stat_item['alg_runtime'] = elapsed_time
        
        if W.size == 0:
            
            
            running_stat_item['failure'] = 1
            
            
            running_stat_item['mse_for_training'] = 99999	
            running_stat_item['mse_for_validation'] = 99999
            
            
            stat_data.append(running_stat_item)
            
            
            str_tmp = "\n\n!!!!!! Skip, Some Exception or Error may have happend when execute least_squares_estimate_linear_regression_alg \n"
            str_tmp += "for X_training_set_Adv = \n" + str(X_training_set_Adv) + "\n\n"
            str_tmp += "and Y_training_set_Adv = \n" + str(Y_training_set_Adv) + "\n\n"
            
            str_tmp += "\n\n\n\n\n\n"
                
            str_tmp += "least_squares_estimate_linear_regression_alg: for word_file_name = " + word_file_name + "\n\n"
        
            str_tmp += "W = \n" + str(W) + "\n\n"
        
            str_tmp += "i_len_word_feature = " + str(i_len_word_feature) + "\n"                
            str_tmp += "b_with_wf = " + str(b_with_wf) + "\n"                
            str_tmp += "b_without_punctuation = " + str(b_without_punctuation) + "\n"
            str_tmp += "b_without_stopwords = " + str(b_without_stopwords) + "\n"
            
            

            str_tmp += "alg runtime = " + str(elapsed_time) + "\n"
         
            
            str_to_write = str_tmp
            
            fout_t2r.write(str_to_write)
            print(str_to_write)            
            
            str_for_statistics += str_tmp
            
            continue
            
        
        
        str_to_write = "lr.least_squares_estimate_linear_regression_alg W = \n" + str(W) + "\n"
        
        
        str_to_write += "least_squares_estimate_linear_regression_alg elapsed_time = " + str(elapsed_time) + "\n"
        
        fout_t2r.write(str_to_write)
        print(str_to_write)
        
        #print("elapsed_time = ", elapsed_time)
        
        
        
        str_tmp = "\n\n\n\n\n\n"
                
        str_tmp += "least_squares_estimate_linear_regression_alg: for word_file_name = " + word_file_name + "\n\n"
        
        str_tmp += "W = \n" + str(W) + "\n\n"
        
        str_tmp += "i_len_word_feature = " + str(i_len_word_feature) + "\n"                
        str_tmp += "b_with_wf = " + str(b_with_wf) + "\n"                
        str_tmp += "b_without_punctuation = " + str(b_without_punctuation) + "\n"
        str_tmp += "b_without_stopwords = " + str(b_without_stopwords) + "\n"
                    

        str_tmp += "alg runtime = " + str(elapsed_time) + "\n"
                
        
        str_for_statistics += str_tmp
        
        str_to_write = str_tmp
        fout_t2r.write(str_to_write)
    
    
        start_time = time.time()
        
        est_Y = np.dot(X_training_set_Adv, W)
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
        
        est_Y = np.dot(X_validation_set_Adv, W)
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
            
            
        stat_data.append(running_stat_item)
    
        
        
    
    print(str_for_statistics)
    
    with open('../t3_result_w_wf_compare_stat.txt','w', buffering=1) as fout_t3rs:
        fout_t3rs.write(str_for_statistics)
        
    with open("../t3_result_w_wf_compare_stat.json", "w") as wf_json_set:
        json.dump(stat_data, wf_json_set)
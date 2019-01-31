# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 02:58:27 2019

"""

import linear_regression as lr

import prepare_features as pf

import numpy as np

import time


    
import json # we need to use the JSON package to load the data, since the data is stored in JSON format


str_for_statistics = ""

stat_data = []

str_to_write = ""

with open('../task3_w_wnf_combined_compare.txt','w', buffering=1) as fout_t2r:

    

    for b_without_punctuation in (False, True):

        for i_word_count_feature_count in (0, 60, 160):
            
            for i_adv_feature_setting in (0, 3):
                        
                #b_without_punctuation = False
                i_adv_feature_power = 0
                

                b_without_stopwords = False
                
                b_with_Advanced_feature = False
                i_Advanced_feature_count = 0
                
                #children^3 term
                #i_adv_feature_setting = 3
                b_adv_feature_replace_original_feature = False 
                
                if i_adv_feature_setting >= 2:
                    b_with_Advanced_feature = True
                    i_Advanced_feature_count = 1
                    i_adv_feature_power = i_adv_feature_setting
                
                
                
                b_with_total_comment_word_number_feature = False
                
                b_with_total_number_of_sentence_feature = False
                
                #add avg words per sentence feature
                b_with_average_word_per_sentence_feature = True
                
                b_with_average_length_per_word_feature = False
                
                
                
                b_with_word_count_feature = False
                #i_word_count_feature_count = 0
                
                word_file_name = "words"
                
                if i_word_count_feature_count > 0:
                    
                    b_with_word_count_feature = True
                    
                    word_file_name += "_" + str(i_word_count_feature_count)
                    
                    if b_without_punctuation:
                        
                        word_file_name += "_wo_punctuation"
                    
                else:
                    
                    word_file_name = "words"
                
                
                
                    
                
                
                
                        
                """ 
                w1, w2, w3, w4 = feature_setting.split('_')
                
                if int(w1) == 1:
                    b_with_total_comment_word_number_feature = True
                    
                if int(w2) == 1:
                    b_with_total_number_of_sentence_feature = True
                    
                if int(w3) == 1:
                    b_with_average_word_per_sentence_feature = True
                    
                if int(w4) == 1:
                    b_with_average_length_per_word_feature = True
                """    
                
                
                
                
                
                
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
                
                
                
                X_testing_set_Adv = np.array([])
                Y_testing_set_Adv = np.array([])
                
            
                start_time = time.time()
            
                # generate feature for training set
                
                #with open("../testing_set.json", "r") as rf_training_set:   
                with open("../training_set.json", "r") as rf_training_set:
                    data_training_set = json.load(rf_training_set)
                    
                
                running_stat_item = {}
                
                    
                X_training_set_Adv, Y_training_set_Adv, str_out = pf.generate_wordfeature_and_output(wordcount, data_training_set, b_with_word_count_feature, i_word_count_feature_count, b_with_Advanced_feature, i_Advanced_feature_count, b_without_punctuation, b_with_total_comment_word_number_feature, b_with_total_number_of_sentence_feature, b_with_average_word_per_sentence_feature, b_with_average_length_per_word_feature, b_without_stopwords, b_adv_feature_replace_original_feature, i_adv_feature_power)
                #X_training_set_Adv, Y_training_set_Adv = pf.generate_wordfeature_and_output(wordcount, data_training_set, b_with_wf, i_len_word_feature, False, 0, b_without_punctuation)
                        
                #print(str_is_root, str_controversiality, str_children, str_popularity_score)
                        
                        
                        
                str_to_write = "\n\n=================!!!!======================"  + "\n"
                
                str_to_write += "X_training_set_Adv = \n" + str(X_training_set_Adv) + "\n" + str(X_training_set_Adv.shape) + "\n"
                
                str_to_write += "Y_training_set_Adv = \n" + str(Y_training_set_Adv) + "\n" + str(Y_training_set_Adv.shape) + "\n"
                
                str_to_write += str_out
                
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
                X_validation_set_Adv, Y_validation_set_Adv, str_out = pf.generate_wordfeature_and_output(wordcount, data_validation_set, b_with_word_count_feature, i_word_count_feature_count, b_with_Advanced_feature, i_Advanced_feature_count, b_without_punctuation, b_with_total_comment_word_number_feature, b_with_total_number_of_sentence_feature, b_with_average_word_per_sentence_feature, b_with_average_length_per_word_feature, b_without_stopwords, b_adv_feature_replace_original_feature, i_adv_feature_power)
                #print(str_is_root, str_controversiality, str_children, str_popularity_score)
                        
                        
                        
                str_to_write = "\n\n=================!!!!======================"  + "\n"
                
                str_to_write += "X_validation_set_Adv = \n" + str(X_validation_set_Adv) + "\n" + str(X_validation_set_Adv.shape) + "\n"
                
                str_to_write += "Y_validation_set_Adv = \n" + str(Y_validation_set_Adv) + "\n" + str(Y_validation_set_Adv.shape) + "\n"
                
                str_to_write += str_out
                
                fout_t2r.write(str_to_write)
                print(str_to_write)
                
                
                elapsed_time = time.time() - start_time
                
                
                running_stat_item['pre_alg_runtime'] += elapsed_time
                
                #print("lr.least_squares_estimate_linear_regression_alg W = \n", W)
                #print("elapsed_time = ", elapsed_time)
                
                str_to_write = "prepare validation set related matrix elapsed_time = " + str(elapsed_time) + "\n\n\n\n\n"
                
                fout_t2r.write(str_to_write)
                print(str_to_write)
                
                        
                
                
                
                
                
                
                
                
                
                
                start_time = time.time()
            
                # generate feature for training set
                
                #with open("../testing_set.json", "r") as rf_training_set:   
                with open("../testing_set.json", "r") as rf_testing_set:
                    data_testing_set = json.load(rf_testing_set)
                    
                  
                #X_validation_set_Adv, Y_validation_set_Adv = pf.generate_wordfeature_and_output(wordcount, data, False, 0, False, 0)
                X_testing_set_Adv, Y_testing_set_Adv, str_out = pf.generate_wordfeature_and_output(wordcount, data_testing_set, b_with_word_count_feature, i_word_count_feature_count, b_with_Advanced_feature, i_Advanced_feature_count, b_without_punctuation, b_with_total_comment_word_number_feature, b_with_total_number_of_sentence_feature, b_with_average_word_per_sentence_feature, b_with_average_length_per_word_feature, b_without_stopwords, b_adv_feature_replace_original_feature, i_adv_feature_power)
                #print(str_is_root, str_controversiality, str_children, str_popularity_score)
                        
                        
                        
                str_to_write = "\n\n=================!!!!======================"  + "\n"
                
                str_to_write += "X_testing_set_Adv = \n" + str(X_testing_set_Adv) + "\n" + str(X_testing_set_Adv.shape) + "\n"
                
                str_to_write += "Y_testing_set_Adv = \n" + str(Y_testing_set_Adv) + "\n" + str(Y_testing_set_Adv.shape) + "\n"
                
                str_to_write += str_out
                
                fout_t2r.write(str_to_write)
                print(str_to_write)
                
                
                elapsed_time = time.time() - start_time
                
                
                running_stat_item['pre_alg_runtime'] += elapsed_time
                
                #print("lr.least_squares_estimate_linear_regression_alg W = \n", W)
                #print("elapsed_time = ", elapsed_time)
                
                str_to_write = "prepare testing set related matrix elapsed_time = " + str(elapsed_time) + "\n\n\n\n\n"
                
                fout_t2r.write(str_to_write)
                print(str_to_write)
                
                
                
                
                
                
                
                
                
               
                
                str_tmp = "for word_file_name = " + word_file_name + "\n\n"
                
                
                str_tmp += "b_without_punctuation = " + str(b_without_punctuation) + "\n"                
                str_tmp += "b_without_stopwords = " + str(b_without_stopwords) + "\n"
                
                
                
                
                str_tmp += "i_word_count_feature_count = " + str(i_word_count_feature_count) + "\n"
                
                
                str_tmp += "b_with_word_count_feature = " + str(b_with_word_count_feature) + "\n"
                
                
                str_tmp += "b_with_total_comment_word_number_feature = " + str(b_with_total_comment_word_number_feature) + "\n"                
                str_tmp += "b_with_total_number_of_sentence_feature = " + str(b_with_total_number_of_sentence_feature) + "\n"                
                str_tmp += "b_with_average_word_per_sentence_feature = " + str(b_with_average_word_per_sentence_feature) + "\n"
                str_tmp += "b_with_average_length_per_word_feature = " + str(b_with_average_length_per_word_feature) + "\n"
                
                
                
                str_tmp += "b_with_Advanced_feature = " + str(b_with_Advanced_feature) + "\n"                
                str_tmp += "b_adv_feature_replace_original_feature = " + str(b_adv_feature_replace_original_feature) + "\n"
                
                str_tmp += "i_adv_feature_power = " + str(i_adv_feature_power) + "\n"
                
                
                
                
                str_to_write = "\n\n\n\n\n\n\n\n" + "for training and validation \n\n ===========!!!!===========\n\n"
                fout_t2r.write(str_to_write)
                print(str_to_write)
                
                print(str_tmp)
                fout_t2r.write(str_tmp)
                
                
                start_time = time.time()
                
                W, str_to_write = lr.least_squares_estimate_linear_regression_alg(X_training_set_Adv, Y_training_set_Adv)
                #print("lr.least_squares_estimate_linear_regression_alg W = \n", W)
                fout_t2r.write(str_to_write)
                print(str_to_write)
                
                elapsed_time = time.time() - start_time
                
                
                running_stat_item['desc'] = "least_squares_estimate_linear_regression_alg"
                        
                running_stat_item['failure'] = 0
                
                
                running_stat_item['alg_runtime'] = elapsed_time
                
                                
                
                running_stat_item['i_word_count_feature_count'] = i_word_count_feature_count 
                
                running_stat_item['b_with_word_count_feature'] = b_with_word_count_feature 
                
                
                
                
                running_stat_item['b_without_punctuation'] = b_without_punctuation                
                running_stat_item['b_without_stopwords'] = b_without_stopwords
                    
                running_stat_item['b_with_total_comment_word_number_feature'] = b_with_total_comment_word_number_feature            
                running_stat_item['b_with_total_number_of_sentence_feature'] = b_with_total_number_of_sentence_feature
                running_stat_item['b_with_average_word_per_sentence_feature'] = b_with_average_word_per_sentence_feature        
                running_stat_item['b_with_average_length_per_word_feature'] = b_with_average_length_per_word_feature
                
                
                
                running_stat_item['b_with_Advanced_feature'] = b_with_Advanced_feature
                
                running_stat_item['b_adv_feature_replace_original_feature'] = b_adv_feature_replace_original_feature
                running_stat_item['i_adv_feature_power'] = i_adv_feature_power
                   
                
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
                    
                    
                    
                    str_tmp += "i_word_count_feature_count = " + str(i_word_count_feature_count) + "\n"
                    
                    str_tmp += "b_with_word_count_feature = " + str(b_with_word_count_feature) + "\n"
                    
                    str_tmp += "b_without_punctuation = " + str(b_without_punctuation) + "\n"                
                    str_tmp += "b_without_stopwords = " + str(b_without_stopwords) + "\n"                    
                
                    str_tmp += "b_with_total_comment_word_number_feature = " + str(b_with_total_comment_word_number_feature) + "\n"                
                    str_tmp += "b_with_total_number_of_sentence_feature = " + str(b_with_total_number_of_sentence_feature) + "\n"                
                    str_tmp += "b_with_average_word_per_sentence_feature = " + str(b_with_average_word_per_sentence_feature) + "\n"
                    str_tmp += "b_with_average_length_per_word_feature = " + str(b_with_average_length_per_word_feature) + "\n"
                    
                    
                    str_tmp += "b_with_Advanced_feature = " + str(b_with_Advanced_feature) + "\n"                    
                    str_tmp += "b_adv_feature_replace_original_feature = " + str(b_adv_feature_replace_original_feature) + "\n"
                    str_tmp += "i_adv_feature_power = " + str(i_adv_feature_power) + "\n"
                
                        
        
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
                
                            
                str_tmp += "i_word_count_feature_count = " + str(i_word_count_feature_count) + "\n"
                
                str_tmp += "b_with_word_count_feature = " + str(b_with_word_count_feature) + "\n"
                
                str_tmp += "b_without_punctuation = " + str(b_without_punctuation) + "\n"                
                str_tmp += "b_without_stopwords = " + str(b_without_stopwords) + "\n"  
            
                str_tmp += "b_with_total_comment_word_number_feature = " + str(b_with_total_comment_word_number_feature) + "\n"                
                str_tmp += "b_with_total_number_of_sentence_feature = " + str(b_with_total_number_of_sentence_feature) + "\n"                
                str_tmp += "b_with_average_word_per_sentence_feature = " + str(b_with_average_word_per_sentence_feature) + "\n"
                str_tmp += "b_with_average_length_per_word_feature = " + str(b_with_average_length_per_word_feature) + "\n"
                
                
                str_tmp += "b_with_Advanced_feature = " + str(b_with_Advanced_feature) + "\n"
                
                str_tmp += "b_adv_feature_replace_original_feature = " + str(b_adv_feature_replace_original_feature) + "\n"
                str_tmp += "i_adv_feature_power = " + str(i_adv_feature_power) + "\n"
                
                
                
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
                
                
                
                
                
                
                
                
                start_time = time.time()
                
                est_Y = np.dot(X_testing_set_Adv, W)
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
                
                
                
                
                
                
                stat_data.append(running_stat_item)
                
                
        
        
        
        
        
    
    print(str_for_statistics)
    
    with open('../task3_w_wnf_combined_compare_stat.txt','w', buffering=1) as fout_t3rs:
        fout_t3rs.write(str_for_statistics)
        
    with open("../task3_w_wnf_combined_compare_stat.json", "w") as wf_json_set:
        json.dump(stat_data, wf_json_set)
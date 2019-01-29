# -*- coding: utf-8 -*-
"""
Created on Sat Jan 26 06:28:33 2019

@author: PeterXu
"""

#import json

import numpy as np

def generate_wordfeature_and_output(dict_wc, data_set, b_with_word_feature=False, i_word_feature_count=0, b_with_Advanced_feature=False, i_Advanced_feature_count=0):
    
    
    # 3 normal feature, 1 extra column
    i_feature_count = 3
    
    if b_with_word_feature:
        i_feature_count += i_word_feature_count
        
    if b_with_Advanced_feature:    
        i_feature_count += i_Advanced_feature_count
    
    i_feature_count += 1
    
    
    output_X = np.array([]).reshape(0, i_feature_count)
    output_Y = np.array([]).reshape(0, 1)
           
    #X_training_set_Adv = np.array([]).reshape(0, 4)
    #Y_training_set_Adv = np.array([]).reshape(0, 1)
    
    for data_point in data_set:
    #for data_point in data_set[:10]:
        
        f_is_root = 0.0
        f_controversiality = 0.0
        f_children = 0.0
        
        f_popularity_score = 0.0
        
        str_is_root = data_point['is_root']
        if str_is_root:
            f_is_root = 1.0
        else:
            f_is_root = 0.0
        
           
            
        str_controversiality = data_point['controversiality']
        f_controversiality = float(str_controversiality)
        
        
        str_children = data_point['children']
        f_children = float(str_children)
        
        #X_training_set = np.array([[, 0.86, 1]])
        
        X_entry = np.array([])
        
        X_entry = np.append(X_entry, [[f_is_root, f_controversiality, f_children]])
    
        if b_with_word_feature:
            str_text = data_point['text']
            
            word_list = str_text.split()
            
            local_wordcount = {}
            
            for word in word_list:
                if word not in local_wordcount:
                    local_wordcount[word] = 1
                else:
                    local_wordcount[word] += 1
                    
            #print("str_text = ", str_text)
            
            wf_X = np.zeros((160, 1))
            #print(X_wf_array)
            #print(X_wf_array.shape)
            
            for key in local_wordcount.keys():
                #print(k, v)
                #str_to_write = k + " " + str(v) + "\n"
                #fout_words.write(str_to_write)
                if key in dict_wc:
                    wf_X[dict_wc[key], 0] = local_wordcount[key]
                    #print(key, wordcount[key], local_wordcount[key])
                    
            #if len(word_list) > 50:
            #print(wf_X)
            #print(wf_X)
            X_entry = np.append(X_entry, [wf_X])
    
            
        
        #if b_with_Advanced_feature:
            
        
        X_entry = np.append(X_entry, [[1.0]])
        #print(X_entry, X_entry.shape)
        #print(output_X, output_X.shape)
        
        output_X = np.append(output_X, [X_entry], axis=0)

        
        str_popularity_score = data_point['popularity_score']
        f_popularity_score = float(str_popularity_score)
        
        output_Y = np.append(output_Y, [[f_popularity_score]], axis=0)

    
    #print("=================================")
    #print(output_X)
    #print(output_Y)
    
    return output_X, output_Y
    #return wf_X, output_Y


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


# generate feature for training set

#with open("../testing_set.json", "r") as rf_training_set:   
with open("../training_set.json", "r") as rf_training_set:
    data = json.load(rf_training_set)
    
  
    generate_wordfeature_and_output(wordcount, data)
    
"""   
            
            
            
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 26 06:28:33 2019

@author: PeterXu
"""

import json

import numpy as np

def generate_wordfeature_and_output(dict_wc, data_set):
    
    for data_point in data_set:
        
        str_text = data_point['text']
        
        word_list = str_text.split()
        
        local_wordcount = {}
        
        for word in word_list:
            if word not in local_wordcount:
                local_wordcount[word] = 1
            else:
                local_wordcount[word] += 1
                
        print("str_text = ", str_text)
        
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

            
        
        output_Y = np.array([[data_point['popularity_score']]])

        print(wf_X)
        print(output_Y)
    
    return
    #return wf_X, output_Y



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
    
   
            
            
            
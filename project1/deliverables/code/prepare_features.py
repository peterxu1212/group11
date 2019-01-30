# -*- coding: utf-8 -*-
"""
Created on Sat Jan 26 06:28:33 2019

@author: PeterXu
"""

#import json

import numpy as np
import re


# common english stop words list, according to http://xpo6.com/list-of-english-stop-words/
tp_eng_stop_words = ("a", "about", "above", "above", "across", "after", "afterwards", "again", "against", "all", "almost", "alone", "along", "already", "also","although","always","am","among", "amongst", "amoungst", "amount",  "an", "and", "another", "any","anyhow","anyone","anything","anyway", "anywhere", "are", "around", "as",  "at", "back","be","became", "because","become","becomes", "becoming", "been", "before", "beforehand", "behind", "being", "below", "beside", "besides", "between", "beyond", "bill", "both", "bottom","but", "by", "call", "can", "cannot", "cant", "co", "con", "could", "couldnt", "cry", "de", "describe", "detail", "do", "done", "down", "due", "during", "each", "eg", "eight", "either", "eleven","else", "elsewhere", "empty", "enough", "etc", "even", "ever", "every", "everyone", "everything", "everywhere", "except", "few", "fifteen", "fify", "fill", "find", "fire", "first", "five", "for", "former", "formerly", "forty", "found", "four", "from", "front", "full", "further", "get", "give", "go", "had", "has", "hasnt", "have", "he", "hence", "her", "here", "hereafter", "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his", "how", "however", "hundred", "ie", "if", "in", "inc", "indeed", "interest", "into", "is", "it", "its", "itself", "keep", "last", "latter", "latterly", "least", "less", "ltd", "made", "many", "may", "me", "meanwhile", "might", "mill", "mine", "more", "moreover", "most", "mostly", "move", "much", "must", "my", "myself", "name", "namely", "neither", "never", "nevertheless", "next", "nine", "no", "nobody", "none", "noone", "nor", "not", "nothing", "now", "nowhere", "of", "off", "often", "on", "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our", "ours", "ourselves", "out", "over", "own","part", "per", "perhaps", "please", "put", "rather", "re", "same", "see", "seem", "seemed", "seeming", "seems", "serious", "several", "she", "should", "show", "side", "since", "sincere", "six", "sixty", "so", "some", "somehow", "someone", "something", "sometime", "sometimes", "somewhere", "still", "such", "system", "take", "ten", "than", "that", "the", "their", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "therefore", "therein", "thereupon", "these", "they", "thickv", "thin", "third", "this", "those", "though", "three", "through", "throughout", "thru", "thus", "to", "together", "too", "top", "toward", "towards", "twelve", "twenty", "two", "un", "under", "until", "up", "upon", "us", "very", "via", "was", "we", "well", "were", "what", "whatever", "when", "whence", "whenever", "where", "whereafter", "whereas", "whereby", "wherein", "whereupon", "wherever", "whether", "which", "while", "whither", "who", "whoever", "whole", "whom", "whose", "why", "will", "with", "within", "without", "would", "yet", "you", "your", "yours", "yourself", "yourselves", "the")



def calc_sentence_number_in_comment(str_comment):

	sentence_count = 0
	seen_end = False
	sentence_end = {'?', '!', '.'}
	for c in str_comment:
		if c in sentence_end:
			if not seen_end:
				seen_end = True
				sentence_count += 1
			continue
		seen_end = False
	
	return sentence_count


def generate_wordfeature_and_output(dict_wc, data_set, b_with_word_count_feature=False, i_word_count_feature_count=0, b_with_Advanced_feature=False, i_Advanced_feature_count=0, b_without_punctuation=False, b_with_total_comment_word_number_feature=False, b_with_total_number_of_sentence_feature=False, b_with_average_word_per_sentence_feature=False, b_with_average_length_per_word_feature=False):
    
    
    # 3 normal feature, 1 extra column
    i_feature_count = 3
    
    if b_with_word_count_feature:
        i_feature_count += i_word_count_feature_count
        
    if b_with_Advanced_feature:    
        i_feature_count += i_Advanced_feature_count
    
    
    if b_with_total_comment_word_number_feature:
        i_feature_count += 1
	
    if b_with_total_number_of_sentence_feature:
        i_feature_count += 1
	
    if b_with_average_word_per_sentence_feature:
        i_feature_count += 1
	
    if b_with_average_length_per_word_feature:
        i_feature_count += 1
	
	
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
    
        if b_with_word_count_feature:
            str_text = data_point['text']
            
            if b_without_punctuation:
                str_text = re.sub(r'[^\w\s]', ' ', str_text)
            
            word_list = str_text.split()
            
            local_wordcount = {}
            
            for word in word_list:
                if word not in local_wordcount:
                    local_wordcount[word] = 1
                else:
                    local_wordcount[word] += 1
                    
            #print("str_text = ", str_text)
            
            wf_X = np.zeros((i_word_count_feature_count, 1))
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
		
        if b_with_total_comment_word_number_feature:

            str_text = data_point['text']
            str_text = re.sub(r'[^\w\s]', ' ', str_text)
            
            word_list = str_text.split()
			
            i_total_comment_word_number = len(word_list)
            X_entry = np.append(X_entry, [[float(i_total_comment_word_number)]])
			
	
        if b_with_total_number_of_sentence_feature:

            str_text = data_point['text']
			
            i_total_number_of_sentence = calc_sentence_number_in_comment(str_text)
            X_entry = np.append(X_entry, [[float(i_total_number_of_sentence)]])
			
	
        if b_with_average_word_per_sentence_feature:
		
            str_text = data_point['text']
			
            i_total_number_of_sentence = calc_sentence_number_in_comment(str_text)
			
            str_text = re.sub(r'[^\w\s]', ' ', str_text)

            word_list = str_text.split()

            i_total_comment_word_number = len(word_list)
            
            if i_total_number_of_sentence == 0:
                i_total_number_of_sentence = 1
			
            f_average_word_per_sentence = float(i_total_comment_word_number) / float(i_total_number_of_sentence)
            X_entry = np.append(X_entry, [[float(f_average_word_per_sentence)]])
			
			
	
        if b_with_average_length_per_word_feature:
			
            str_text = data_point['text']
            str_text = re.sub(r'[^\w\s]', ' ', str_text)
			
            word_list = str_text.split()

            i_total_comment_word_number = len(word_list)
			
            i_total_number_of_letters = 0
			
            for per_word in word_list:
                i_total_number_of_letters += len(per_word)
			
            if i_total_comment_word_number == 0:
                i_total_comment_word_number = 1            
            
            f_average_length_per_word = float(i_total_number_of_letters) / float(i_total_comment_word_number)
            
            
            
            X_entry = np.append(X_entry, [[float(f_average_length_per_word)]])
            
			
        
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
            
            
            
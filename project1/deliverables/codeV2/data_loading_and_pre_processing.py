
import json

import re

import prepare_features as pf



# pre-process text field to lower case
# And generate word count files

# split the the raw data into training set, validation set and testing set


with open("../proj1_data.json", "r") as read_file:
    data = json.load(read_file)
    
    



str_output = ""

wordcount = {}

wordcount_wo_punctuation = {}

wordcount_wo_sw = {}

wordcount_wo_punctuation_and_sw = {}
    
for data_point in data:
    str_text = data_point['text']
    str_text_lc = str_text.lower()
    
       
    word_list = str_text_lc.split()
    
    
    for word in word_list:
        if word not in wordcount:
            wordcount[word] = 1
        else:
            wordcount[word] += 1
    
    
    
    word_list_without_sw = pf.remove_stop_words(word_list)
    
    for word_wo_sw in word_list_without_sw:
        if word_wo_sw not in wordcount_wo_sw:
            wordcount_wo_sw[word_wo_sw] = 1
        else:
            wordcount_wo_sw[word_wo_sw] += 1
      
    
    
    
    
    #deal with punctuation marks    
    str_text_lc_without_punctuation = re.sub(r'[^\w\s]', ' ', str_text_lc)   
 
    
    word_list_without_punctuation = str_text_lc_without_punctuation.split()
    
    for word_wo_punctation in word_list_without_punctuation:
        if word_wo_punctation not in wordcount_wo_punctuation:
            wordcount_wo_punctuation[word_wo_punctation] = 1
        else:
            wordcount_wo_punctuation[word_wo_punctation] += 1
       
    
    
    word_list_wo_punctuation_and_sw = pf.remove_stop_words(word_list_without_punctuation)
    
    for word_wo_punctation_and_sw in word_list_wo_punctuation_and_sw:
        if word_wo_punctation_and_sw not in wordcount_wo_punctuation_and_sw:
            wordcount_wo_punctuation_and_sw[word_wo_punctation_and_sw] = 1
        else:
            wordcount_wo_punctuation_and_sw[word_wo_punctation_and_sw] += 1
    
    
    data_point['text'] = str_text_lc








wordcount = sorted(wordcount.items(), key = lambda x:x[1], reverse=True)

with open('../words.txt','w') as fout_words:
    for k, v in wordcount[:160]:
        print(k, v)
        str_to_write = k + " " + str(v) + "\n"
        fout_words.write(str_to_write)

for list_size in (300, 260, 160, 60):
    words_file_name = '../words_' + str(list_size) + '.txt'
    
    with open(words_file_name,'w') as fout_words:
        for k, v in wordcount[:list_size]:
            print(k, v)
            str_to_write = k + " " + str(v) + "\n"
            fout_words.write(str_to_write)



wordcount_wo_punctuation = sorted(wordcount_wo_punctuation.items(), key = lambda x:x[1], reverse=True)

for list_size in (300, 260, 160, 60):
    words_file_name = '../words_' + str(list_size) + '_wo_punctuation' + '.txt'
    
    with open(words_file_name,'w') as fout_words:
        for k, v in wordcount_wo_punctuation[:list_size]:
            print(k, v)
            str_to_write = k + " " + str(v) + "\n"
            fout_words.write(str_to_write)


wordcount_wo_sw = sorted(wordcount_wo_sw.items(), key = lambda x:x[1], reverse=True)

for list_size in (300, 260, 160, 60):
    words_file_name = '../words_' + str(list_size) + '_wo_stopwords' + '.txt'
    
    with open(words_file_name,'w') as fout_words:
        for k, v in wordcount_wo_sw[:list_size]:
            print(k, v)
            str_to_write = k + " " + str(v) + "\n"
            fout_words.write(str_to_write)



wordcount_wo_punctuation_and_sw = sorted(wordcount_wo_punctuation_and_sw.items(), key = lambda x:x[1], reverse=True)

for list_size in (300, 260, 160, 60):
    words_file_name = '../words_' + str(list_size) + '_adv' + '.txt'
    
    with open(words_file_name,'w') as fout_words:
        for k, v in wordcount_wo_punctuation_and_sw[:list_size]:
            print(k, v)
            str_to_write = k + " " + str(v) + "\n"
            fout_words.write(str_to_write)




# Split the data

i_len_data = len(data)

training_set = data[0:10000]

i_len_training_set = len(training_set)

validation_set = data[10000:11000]

i_len_validation_set = len(validation_set)

testing_set = data[11000:12000]

i_len_testing_set = len(testing_set)


print("i_len_data = ", i_len_data)

print("i_len_training_set = ", i_len_training_set)

print("i_len_validation_set = ", i_len_validation_set)

print("i_len_testing_set = ", i_len_testing_set)

with open("../training_set.json", "w") as wf_training_set:
    json.dump(training_set, wf_training_set)
    
with open("../validation_set.json", "w") as wf_validation_set:
    json.dump(validation_set, wf_validation_set)
    
with open("../testing_set.json", "w") as wf_testing_set:
    json.dump(testing_set, wf_testing_set)


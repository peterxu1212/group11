
import json # we need to use the JSON package to load the data, since the data is stored in JSON format

import re


with open("../proj1_data.json", "r") as read_file:
    data = json.load(read_file)
    
    
# Now the data is loaded.
# It a list of data points, where each datapoint is a dictionary with the following attributes:
# popularity_score : a popularity score for this comment (based on the number of upvotes) (type: float)
# children : the number of replies to this comment (type: int)
# text : the text of this comment (type: string)
# controversiality : a score for how "controversial" this comment is (automatically computed by Reddit)
# is_root : if True, then this comment is a direct reply to a post; if False, this is a direct reply to another comment 


# pre-process text field to lower case
# And generate word counts



str_output = ""

wordcount = {}

wordcount_without_punctuation = {}
    
for data_point in data:
    str_text = data_point['text']
    str_text_lc = str_text.lower()
    
       
    word_list = str_text_lc.split()
    
    
    for word in word_list:
        if word not in wordcount:
            wordcount[word] = 1
        else:
            wordcount[word] += 1
    
    #***deal with punctuation marks?    
    str_text_lc_without_punctuation = re.sub(r'[^\w\s]', ' ', str_text_lc)
    
    #if str_text_lc_without_punctuation != str_text_lc:
        #print(str_text_lc_without_punctuation, " is different from ", str_text_lc)
        #str_output += 
    
    word_list_without_punctuation = str_text_lc_without_punctuation.split()
    
    for word_wo_punctation in word_list_without_punctuation:
        if word_wo_punctation not in wordcount_without_punctuation:
            wordcount_without_punctuation[word_wo_punctation] = 1
        else:
            wordcount_without_punctuation[word_wo_punctation] += 1
    
    
    data_point['text'] = str_text_lc


wordcount = sorted(wordcount.items(), key = lambda x:x[1], reverse=True)

with open('../words.txt','w') as fout_words:
    for k, v in wordcount[:160]:
        print(k, v)
        str_to_write = k + " " + str(v) + "\n"
        fout_words.write(str_to_write)


with open('../words_300.txt','w') as fout_words:
    for k, v in wordcount[:300]:
        print(k, v)
        str_to_write = k + " " + str(v) + "\n"
        fout_words.write(str_to_write)
        
with open('../words_260.txt','w') as fout_words:
    for k, v in wordcount[:260]:
        print(k, v)
        str_to_write = k + " " + str(v) + "\n"
        fout_words.write(str_to_write)


with open('../words_160.txt','w') as fout_words:
    for k, v in wordcount[:160]:
        print(k, v)
        str_to_write = k + " " + str(v) + "\n"
        fout_words.write(str_to_write)

with open('../words_60.txt','w') as fout_words:
    for k, v in wordcount[:60]:
        print(k, v)
        str_to_write = k + " " + str(v) + "\n"
        fout_words.write(str_to_write)


wordcount_without_punctuation = sorted(wordcount_without_punctuation.items(), key = lambda x:x[1], reverse=True)

with open('../words_300_without_punctuation.txt','w') as fout_words_wo_punctuation:
    for k, v in wordcount_without_punctuation[:300]:
        print(k, v)
        str_to_write = k + " " + str(v) + "\n"
        fout_words_wo_punctuation.write(str_to_write)


with open('../words_260_without_punctuation.txt','w') as fout_words_wo_punctuation:
    for k, v in wordcount_without_punctuation[:260]:
        print(k, v)
        str_to_write = k + " " + str(v) + "\n"
        fout_words_wo_punctuation.write(str_to_write)



with open('../words_160_without_punctuation.txt','w') as fout_words_wo_punctuation:
    for k, v in wordcount_without_punctuation[:160]:
        print(k, v)
        str_to_write = k + " " + str(v) + "\n"
        fout_words_wo_punctuation.write(str_to_write)


with open('../words_60_without_punctuation.txt','w') as fout_words_wo_punctuation:
    for k, v in wordcount_without_punctuation[:60]:
        print(k, v)
        str_to_write = k + " " + str(v) + "\n"
        fout_words_wo_punctuation.write(str_to_write)


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


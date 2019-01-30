
import json # we need to use the JSON package to load the data, since the data is stored in JSON format



with open("../training_set.json", "r") as read_file:
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

"""
wordcount = {}

wordcount_without_punctuation = {}
"""
    
data_set_partial = data[:100]

print(data_set_partial)

print(type(data_set_partial))



for data_point in data_set_partial:
    #str_text = data_point['text']
    #str_text_lc = str_text.lower()
    
    #data_point['text'] = str_text_lc
    print(data_point, "\n\n")

#L.sort(key=lambda x:(x[1],x[0]))
data_set_partial_sorted = sorted(data_set_partial, key=lambda x: (x['children'], x['popularity_score']), reverse=True)

for data_point in data_set_partial_sorted:
    print(data_point, "\n\n")


# add python plot code here

#print("i_len_data = ")
    
with open("../tmp_set.json", "w") as wf_testing_set:
    json.dump(data_set_partial, wf_testing_set)


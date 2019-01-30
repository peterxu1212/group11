
import json # we need to use the JSON package to load the data, since the data is stored in JSON format



with open("../t3_result_w_wf_compare_stat.json", "r") as read_file:
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
    
data_set_partial = data

#print(data_set_partial)

print(type(data_set_partial), len(data_set_partial))


#data_set_partial_sorted = sorted(data_set_partial, key=lambda x: x['mse_for_validation'], reverse=True)

"""
data_set_least_squares_alg = [item for item in data_set_partial_sorted if item['desc'] == "least_squares_estimate_linear_regression_alg"]

data_set_gd_alg = [item for item in data_set_partial_sorted if item['desc'] == "gradient_descent_linear_regression_alg"]

data_set_gd_alg_old = [item for item in data_set_partial_sorted if item['desc'] == "gradient_descent_linear_regression_alg_old"]

print("data_set_partial_sorted len = ", len(data_set_partial_sorted))


print("data_set_least_squares_alg len = ", len(data_set_least_squares_alg))


print("data_set_gd_alg len = ", len(data_set_gd_alg))
"""

print("data_set_partial len = ", len(data_set_partial))

#for data_point in data_set_partial:
    #str_text = data_point['text']
    #str_text_lc = str_text.lower()
    
    #data_point['text'] = str_text_lc
    #print(data_point, "\n\n")

#L.sort(key=lambda x:(x[1],x[0]))


#data_set_partial_sorted = sorted(data_set_partial, key=lambda x: (x['desc'], x['popularity_score'], ), reverse=True)


data_set_partial_sorted = sorted(data_set_partial, key=lambda x: (x['mse_for_validation'], x['i_word_count_feature_count']), reverse=True)

#data_set_gd_alg_sorted = sorted(data_set_gd_alg, key=lambda x: (x['mse_for_weight_vector'], x['epsilon_power'], x['distance_rate_power'], x['eta_power']), reverse=True)

"""
for data_point in data_set_gd_alg_old_sorted:
    print(data_point['mse_for_weight_vector'], data_point['epsilon_power'], data_point['eta_power'], "\n", data_point['W'], data_point['T_Robbins_Monroe_pow'], "\n\n")
""" 


for data_point in data_set_partial_sorted:
    #print(data_point['mse_for_validation'], data_point['b_with_average_word_per_sentence_feature'], data_point['distance_rate_power'], data_point['eta_power'], "\n", data_point['W'], data_point['T_Robbins_Monroe_pow'], "\n\n")
    print(data_point, "\n\n")
    #pass


# add python plot code here

#print("i_len_data = ")
    
with open("../tmp_set.json", "w") as wf_testing_set:
    json.dump(data_set_partial, wf_testing_set)


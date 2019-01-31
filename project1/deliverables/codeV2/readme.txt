
task #1:

    execute data_loading_and_pre_processing.py -- To split the reddit raw data into training set, validation set and testing set. Also,   pre-process text field to lower case. In addition, generate word count files, for we not only support the basic method that split by white space, but also deal with punctuation removing and filter out stop words.
    
    prepare_features.py -- implemented a function "generate_wordfeature_and_output" to support generate word related features, include the basic high frequency word count feature and also some other word related features, such as total comment word number feature (how many words in a specific comment), total number of sentences in the comment, average number of words per sentence, and  average length (in letter) per word. 
    
task #2:
    
    implemented in linear_regression.py -- support the closed form (least_squares_estimate_linear_regression_alg) algorithm, and the normal version of the gradient descent algorithm according to the project requirement. In addition, an improved version of gradient descent algorithm is also implemented, by introducing a new parameter called "distance_rate", which helps to let the learning rate value keep a reasonable distance from the epsilon value, so that the learning rate would not decay too quick. Therefore, this distance_rate could help the algorithm run a reasonable enough time in order to improve the accuracy of the result (evaluated in MSE). This distance_rate cooperates well with the Robbins Monroe logic, so could assist to achieve a well balance between run-time and accuracy.
    
        
task #3:

    execute task3_compare_algs.py -- for the comparsion between closed form alg and two gradient descent algs with vary hyperparameters
    
    execute task3_w_wf_compare.py -- for the comparsion between the model with high frequency word count feature and the model without
    
    execute task3_w_wf_additional_compare.py -- to evaluate the performances of new added features


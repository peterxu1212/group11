# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 02:58:27 2019
@author: PeterXu
"""

import linear_regression4 as lr

import prepare_features as pf

import numpy as np

import time



import json # we need to use the JSON package to load the data, since the data is stored in JSON format

import matplotlib.pyplot as plt #畫圖用的

#def

str_for_statistics = ""

str_to_write = ""

with open('../t3_results.txt','w', buffering=1) as fout_t2r:

    #str_to_write = k + " " + str(v) + "\n"
    #fout_t2r.write(str_to_write)

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



    X_training_set_Adv = np.array([])
    Y_training_set_Adv = np.array([])


    X_validation_set_Adv = np.array([])
    Y_validation_set_Adv = np.array([])

    start_time = time.time()

    # generate feature for training set

    #with open("../testing_set.json", "r") as rf_training_set:
    with open("../training_set.json", "r") as rf_training_set:
        data = json.load(rf_training_set)


    X_training_set_Adv, Y_training_set_Adv = pf.generate_wordfeature_and_output(wordcount, data, False, 0, False, 0)


    #X_training_set_Adv, Y_training_set_Adv = pf.generate_wordfeature_and_output(wordcount, data, True, 160, False, 0)



        #print(str_is_root, str_controversiality, str_children, str_popularity_score)



    str_to_write = "\n\n=================!!!!======================"  + "\n"

    str_to_write += "X_training_set_Adv = \n" + str(X_training_set_Adv) + "\n" + str(X_training_set_Adv.shape) + "\n"

    str_to_write += "Y_training_set_Adv = \n" + str(Y_training_set_Adv) + "\n" + str(Y_training_set_Adv.shape) + "\n"

    fout_t2r.write(str_to_write)
    print(str_to_write)


    elapsed_time = time.time() - start_time
    #print("lr.least_squares_estimate_linear_regression_alg W = \n", W)
    #print("elapsed_time = ", elapsed_time)

    str_to_write = "prepare training set related matrix elapsed_time = " + str(elapsed_time) + "\n\n\n\n\n"

    fout_t2r.write(str_to_write)
    print(str_to_write)



    start_time = time.time()

    # generate feature for training set

    #with open("../testing_set.json", "r") as rf_training_set:
    with open("../validation_set.json", "r") as rf_validation_set:
        data = json.load(rf_validation_set)


    X_validation_set_Adv, Y_validation_set_Adv = pf.generate_wordfeature_and_output(wordcount, data, False, 0, False, 0)


    #X_training_set_Adv, Y_training_set_Adv = pf.generate_wordfeature_and_output(wordcount, data, True, 160, False, 0)



        #print(str_is_root, str_controversiality, str_children, str_popularity_score)



    str_to_write = "\n\n=================!!!!======================"  + "\n"

    str_to_write += "X_validation_set_Adv = \n" + str(X_validation_set_Adv) + "\n" + str(X_validation_set_Adv.shape) + "\n"

    str_to_write += "Y_validation_set_Adv = \n" + str(Y_validation_set_Adv) + "\n" + str(Y_validation_set_Adv.shape) + "\n"

    fout_t2r.write(str_to_write)
    print(str_to_write)


    elapsed_time = time.time() - start_time
    #print("lr.least_squares_estimate_linear_regression_alg W = \n", W)
    #print("elapsed_time = ", elapsed_time)

    str_to_write = "prepare validation set related matrix elapsed_time = " + str(elapsed_time) + "\n\n\n\n\n"

    fout_t2r.write(str_to_write)
    print(str_to_write)




    """

    start_time = time.time()
    # your code

    W, str_to_write = lr.least_squares_estimate_linear_regression_alg(X, Y)

    fout_t2r.write(str_to_write)
    print(str_to_write)

    elapsed_time = time.time() - start_time
    #print("lr.least_squares_estimate_linear_regression_alg W = \n", W)
    #print("elapsed_time = ", elapsed_time)

    str_to_write = "lr.least_squares_estimate_linear_regression_alg W = \n" + str(W) + "\n"
    str_to_write += "elapsed_time = " + str(elapsed_time) + "\n\n\n\n\n"

    fout_t2r.write(str_to_write)
    print(str_to_write)


    start_time = time.time()

    W, str_to_write = lr.gradient_descent_linear_regression_alg(X, Y)

    fout_t2r.write(str_to_write)
    print(str_to_write)

    elapsed_time = time.time() - start_time
    #print("lr.gradient_descent_linear_regression_alg W = \n", W)
    #print("elapsed_time = ", elapsed_time)

    str_to_write = "lr.gradient_descent_linear_regression_alg W = \n" + str(W) + "\n"
    str_to_write += "elapsed_time = " + str(elapsed_time) + "\n"

    fout_t2r.write(str_to_write)
    print(str_to_write)

    """










    str_to_write = "\n\n\n\n\n\n\n\n" + "for training and validation \n\n ===========!!!!===========\n\n"
    fout_t2r.write(str_to_write)
    print(str_to_write)


    start_time = time.time()

    W, str_to_write = lr.least_squares_estimate_linear_regression_alg(X_training_set_Adv, Y_training_set_Adv)
    #print("lr.least_squares_estimate_linear_regression_alg W = \n", W)
    fout_t2r.write(str_to_write)
    print(str_to_write)


    str_to_write = "lr.least_squares_estimate_linear_regression_alg W = \n" + str(W) + "\n"

    elapsed_time = time.time() - start_time
    str_to_write += "least_squares_estimate_linear_regression_alg elapsed_time = " + str(elapsed_time) + "\n"

    fout_t2r.write(str_to_write)
    print(str_to_write)

    #print("elapsed_time = ", elapsed_time)

    str_for_statistics += "least_squares_estimate_linear_regression_alg:" + "\n\n"

    str_for_statistics += "W = \n" + str(W) + "\n\n"

    str_for_statistics += "alg runtime = " + str(elapsed_time) + "\n"




    start_time = time.time()

    est_Y = np.dot(X_training_set_Adv, W)
    trn_mse_closed, diff_AB, Sigma_Square_of_diff_AB = lr.mean_squared_error(est_Y, Y_training_set_Adv)

    #print("est_Y = \n", est_Y)
    #print("tmp_mse =", tmp_mse)

    elapsed_time = time.time() - start_time




    str_to_write += "mean_squared_error elapsed_time = " + str(elapsed_time) + "\n"


    Sigma_Square_of_diff_AB = np.sort(Sigma_Square_of_diff_AB, axis=None, kind='mergesort')
    str_to_write += "Sigma_Square_of_diff_AB = \n" + str(Sigma_Square_of_diff_AB) + "\n" + str(Sigma_Square_of_diff_AB.shape) + "\n\n"


    str_to_write += "diff_AB = \n" + str(diff_AB) + "\n" + str(diff_AB.shape) + "\n\n\n\n"

    str_to_write += "est_Y = \n" + str(est_Y) + "\n" + str(est_Y.shape) + "\n"

    fout_t2r.write(str_to_write)
    print(str_to_write)








    start_time = time.time()

    est_Y = np.dot(X_validation_set_Adv, W)
    val_mse_closed, diff_AB, Sigma_Square_of_diff_AB = lr.mean_squared_error(est_Y, Y_validation_set_Adv)

    #print("est_Y = \n", est_Y)
    #print("tmp_mse =", tmp_mse)

    elapsed_time = time.time() - start_time




    str_to_write += "mean_squared_error elapsed_time = " + str(elapsed_time) + "\n"


    Sigma_Square_of_diff_AB = np.sort(Sigma_Square_of_diff_AB, axis=None, kind='mergesort')
    str_to_write += "Sigma_Square_of_diff_AB = \n" + str(Sigma_Square_of_diff_AB) + "\n" + str(Sigma_Square_of_diff_AB.shape) + "\n\n"


    str_to_write += "diff_AB = \n" + str(diff_AB) + "\n" + str(diff_AB.shape) + "\n\n\n\n"

    str_to_write += "est_Y = \n" + str(est_Y) + "\n" + str(est_Y.shape) + "\n"

    fout_t2r.write(str_to_write)
    print(str_to_write)





    # report runtime, MSE and weights, and (stability), w.r.t differet hyperparameters


    #epsilon pow: -3 ~ -8

    #eta pow: 0 ~ -3

    #DR pow: 2 ~ 5

    #TRM pow: 1 ~ 6

    W_closed, str_to_write = lr.least_squares_estimate_linear_regression_alg(X_validation_set_Adv, Y_validation_set_Adv)

    list_T_Robbins_Monroe_pow = []
    list_trn_mse = []
    list_val_mse = []
    list_elapsed_time = []
    list_W_MSE = []

    epsilon_power = -6

    # 0 ~ -3
    eta_power = 0
    distance_rate_power = 4

    for T_Robbins_Monroe_pow in range(1, 8, 1):
        list_T_Robbins_Monroe_pow.append(T_Robbins_Monroe_pow)

        start_time = time.time()

        W = lr.gradient_descent_linear_regression_alg_old(X_training_set_Adv, Y_training_set_Adv, 10**epsilon_power, 10**eta_power, 10**T_Robbins_Monroe_pow)


        elapsed_time = time.time() - start_time

        list_elapsed_time.append(elapsed_time)

        W_MSE, Q1, Q2 = lr.mean_squared_error(W, W_closed)
        list_W_MSE.append(W_MSE)

        est_Y = np.dot(X_validation_set_Adv, W)
        val_mse, diff_AB, Sigma_Square_of_diff_AB = lr.mean_squared_error(est_Y, Y_validation_set_Adv)
        est_Y = np.dot(X_training_set_Adv, W)
        trn_mse, diff_AB, Sigma_Square_of_diff_AB = lr.mean_squared_error(est_Y, Y_training_set_Adv)
        list_trn_mse.append(trn_mse)
        list_val_mse.append(val_mse)

    plt.plot(list_T_Robbins_Monroe_pow, list_val_mse, "bo")
    plt.plot([1, 7],[val_mse_closed, val_mse_closed])
    plt.title("Plot of validation MSE w.r.t log T")
    plt.legend(["Gradient descent validation MSE",'Closed form MSE'])
    plt.show()
    plt.plot(list_T_Robbins_Monroe_pow, list_W_MSE, "bo")
    plt.title("Plot of MSE of W w.r.t log T")
    plt.show()
    plt.plot(list_T_Robbins_Monroe_pow, list_elapsed_time, "bo")
    plt.title("Plot of running time w.r.t log T")
    plt.show()
    #這裏running time會抽風

    list_epsilon_power = []
    list_trn_mse = []
    list_val_mse = []
    list_elapsed_time = []
    list_W_MSE = []

    epsilon_power = -6

    # 0 ~ -3
    eta_power = 0
    distance_rate_power = 4
    T_Robbins_Monroe_pow = 2

    for epsilon_power in range(-3, -8, -1):
        list_epsilon_power.append(epsilon_power)

        start_time = time.time()

        W = lr.gradient_descent_linear_regression_alg_old(X_training_set_Adv, Y_training_set_Adv, 10**epsilon_power, 10**eta_power, 10**T_Robbins_Monroe_pow)


        elapsed_time = time.time() - start_time

        list_elapsed_time.append(elapsed_time)

        W_MSE, Q1, Q2 = lr.mean_squared_error(W, W_closed)
        list_W_MSE.append(W_MSE)

        est_Y = np.dot(X_validation_set_Adv, W)
        val_mse, diff_AB, Sigma_Square_of_diff_AB = lr.mean_squared_error(est_Y, Y_validation_set_Adv)
        est_Y = np.dot(X_training_set_Adv, W)
        trn_mse, diff_AB, Sigma_Square_of_diff_AB = lr.mean_squared_error(est_Y, Y_training_set_Adv)
        list_trn_mse.append(trn_mse)
        list_val_mse.append(val_mse)

    plt.plot(list_epsilon_power, list_val_mse, "bo")
    plt.plot([-3, -7],[val_mse_closed, val_mse_closed])
    plt.title("Plot of validation MSE w.r.t log epsilon")
    plt.legend(["Gradient descent validation MSE",'Closed form MSE'])
    plt.show()
    plt.plot(list_epsilon_power, list_W_MSE, "bo")
    plt.title("Plot of MSE of W w.r.t log epsilon")
    plt.show()
    plt.plot(list_epsilon_power, list_elapsed_time, "bo")
    plt.title("Plot of running time w.r.t log epsilon")
    plt.show()

    list_eta_power = []
    list_trn_mse = []
    list_val_mse = []
    list_elapsed_time = []
    list_W_MSE = []

    epsilon_power = -5
    T_Robbins_Monroe_pow = 2

    for eta_power in range(0, -4, -1):
        list_eta_power.append(eta_power)

        start_time = time.time()

        W = lr.gradient_descent_linear_regression_alg_old(X_training_set_Adv, Y_training_set_Adv, 10**epsilon_power, 10**eta_power, 10**T_Robbins_Monroe_pow)


        elapsed_time = time.time() - start_time

        list_elapsed_time.append(elapsed_time)

        W_MSE, Q1, Q2 = lr.mean_squared_error(W, W_closed)
        list_W_MSE.append(W_MSE)

        est_Y = np.dot(X_validation_set_Adv, W)
        val_mse, diff_AB, Sigma_Square_of_diff_AB = lr.mean_squared_error(est_Y, Y_validation_set_Adv)
        est_Y = np.dot(X_training_set_Adv, W)
        trn_mse, diff_AB, Sigma_Square_of_diff_AB = lr.mean_squared_error(est_Y, Y_training_set_Adv)
        list_trn_mse.append(trn_mse)
        list_val_mse.append(val_mse)

    plt.plot(list_eta_power, list_val_mse, "bo")
    plt.plot([0, -3],[val_mse_closed, val_mse_closed])
    plt.title("Plot of validation MSE w.r.t log eta")
    plt.legend(["Gradient descent validation MSE",'Closed form MSE'])
    plt.show()
    plt.plot(list_eta_power, list_W_MSE, "bo")
    plt.title("Plot of MSE of W w.r.t log eta")
    plt.show()
    plt.plot(list_eta_power, list_elapsed_time, "bo")
    plt.title("Plot of running time w.r.t log eta")
    plt.show()

    epsilon_power = -5
    eta_power = 0
    T_Robbins_Monroe_pow = 2

    start_time = time.time()
    W = lr.gradient_descent_linear_regression_alg_old(X_training_set_Adv, Y_training_set_Adv, 10**epsilon_power, 10**eta_power, 10**T_Robbins_Monroe_pow)
    old_W_MSE, Q1, Q2 = lr.mean_squared_error(W, W_closed)
    old_elapsed_time = time.time() - start_time

    est_Y = np.dot(X_validation_set_Adv, W)
    old_MSE, Q1, Q2 = lr.mean_squared_error(est_Y, Y_validation_set_Adv)


    list_distance_rate_power = []
    list_trn_mse = []
    list_val_mse = []
    list_elapsed_time = []
    list_W_MSE = []

    for distance_rate_power in range(2, 8, 1):

        list_distance_rate_power.append(distance_rate_power)

        start_time = time.time()

        W = lr.gradient_descent_linear_regression_alg(X_training_set_Adv, Y_training_set_Adv, 10**epsilon_power, 10**eta_power, distance_rate_power, 10**T_Robbins_Monroe_pow)


        elapsed_time = time.time() - start_time

        list_elapsed_time.append(elapsed_time)

        W_MSE, Q1, Q2 = lr.mean_squared_error(W, W_closed)
        list_W_MSE.append(W_MSE)

        est_Y = np.dot(X_validation_set_Adv, W)
        val_mse, diff_AB, Sigma_Square_of_diff_AB = lr.mean_squared_error(est_Y, Y_validation_set_Adv)
        est_Y = np.dot(X_training_set_Adv, W)
        trn_mse, diff_AB, Sigma_Square_of_diff_AB = lr.mean_squared_error(est_Y, Y_training_set_Adv)
        list_trn_mse.append(trn_mse)
        list_val_mse.append(val_mse)

    plt.plot(list_distance_rate_power, list_val_mse, "bo")
    plt.plot([2, 7],[val_mse_closed, val_mse_closed])
    plt.plot([2, 7],[old_MSE, old_MSE])
    plt.title("Plot of validation MSE w.r.t log DRP")
    plt.legend(["New gradient descent validation MSE",'Closed form MSE', 'Gradient descent MSE'])
    plt.show()
    plt.plot(list_distance_rate_power, list_W_MSE, "bo")
    plt.plot([2, 7],[old_W_MSE, old_W_MSE])
    plt.legend(["New gradient descent MSE of W", 'Gradient descent MSE of W'])
    plt.title("Plot of MSE of W w.r.t log eta")
    plt.show()
    plt.plot(list_distance_rate_power, list_elapsed_time, "bo")
    plt.plot([2,7],[old_elapsed_time, old_elapsed_time])
    plt.title("Plot of running time w.r.t log log DRP")
    plt.legend(["New gradient descent validation running time", 'Gradient descent running time'])
    plt.show()

# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 02:58:27 2019
@author: PeterXu
"""

import linear_regression as lr

import prepare_features as pf

import numpy as np

import time

import matplotlib.pyplot as plt

import json # we need to use the JSON package to load the data, since the data is stored in JSON format

#導入training set和validation set
wordcount = {}

wc_index = 0

X_training_set_Adv = np.array([])
Y_training_set_Adv = np.array([])


X_validation_set_Adv = np.array([])
Y_validation_set_Adv = np.array([])


running_stat_item = {}

with open("../training_set.json", "r") as rf_training_set:
    data = json.load(rf_training_set)

X_training_set_Adv, Y_training_set_Adv, str_output = pf.generate_wordfeature_and_output(wordcount, data, False, 0, False, 0)

with open("../validation_set.json", "r") as rf_validation_set:
    data = json.load(rf_validation_set)


X_validation_set_Adv, Y_validation_set_Adv, str_output = pf.generate_wordfeature_and_output(wordcount, data, False, 0, False, 0)

#測試Closed form的性能
start_time = time.time()

W_closed, str_to_write = lr.least_squares_estimate_linear_regression_alg(X_validation_set_Adv, Y_validation_set_Adv)

closed_running_time = time.time() - start_time

est_Y = np.dot(X_training_set_Adv, W_closed)
trn_mse_closed, diff_AB, Sigma_Square_of_diff_AB = lr.mean_squared_error(est_Y, Y_training_set_Adv)

est_Y = np.dot(X_validation_set_Adv, W_closed)
val_mse_closed, diff_AB, Sigma_Square_of_diff_AB = lr.mean_squared_error(est_Y, Y_validation_set_Adv)

#測試Gradient Descent的性能
list_T_Robbins_Monroe_pow = []
list_trn_mse = []
list_val_mse = []
list_elapsed_time = []
list_W_MSE = []

epsilon_power = -6
eta_power = 0

for T_Robbins_Monroe_pow in range(1, 8, 1):
    list_T_Robbins_Monroe_pow.append(T_Robbins_Monroe_pow)

    start_time = time.time()

    W, Q1, Q2 = lr.gradient_descent_linear_regression_alg_old(X_training_set_Adv, Y_training_set_Adv, 10**epsilon_power, 10**eta_power, 10**T_Robbins_Monroe_pow)


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
plt.xlabel("log T")
plt.ylabel("MSE")
plt.title("Plot of validation MSE w.r.t log T")
plt.legend(["Gradient descent validation MSE",'Closed form MSE'])
plt.show()
plt.plot(list_T_Robbins_Monroe_pow, list_W_MSE, "bo")
plt.xlabel("log T")
plt.ylabel("MSE of W")
plt.title("Plot of MSE of W w.r.t log T")
plt.show()
plt.plot(list_T_Robbins_Monroe_pow, list_elapsed_time, "bo")
plt.plot([1,7],[closed_running_time, closed_running_time])
plt.xlabel("log T")
plt.ylabel("Running time")
plt.title("Plot of running time w.r.t log T")
plt.legend(["Gradient descent validation running time",'Closed form running time'])
plt.show()
#這裏running time會抽風

list_epsilon_power = []
list_trn_mse = []
list_val_mse = []
list_elapsed_time = []
list_W_MSE = []

epsilon_power = -6
eta_power = 0
T_Robbins_Monroe_pow = 2

for epsilon_power in range(-3, -8, -1):
    list_epsilon_power.append(epsilon_power)

    start_time = time.time()

    W, Q1, Q2 = lr.gradient_descent_linear_regression_alg_old(X_training_set_Adv, Y_training_set_Adv, 10**epsilon_power, 10**eta_power, 10**T_Robbins_Monroe_pow)


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
plt.xlabel("log epsilon")
plt.ylabel("MSE")
plt.title("Plot of validation MSE w.r.t log epsilon")
plt.legend(["Gradient descent validation MSE",'Closed form MSE'])
plt.show()
plt.plot(list_epsilon_power, list_W_MSE, "bo")
plt.xlabel("log epsilon")
plt.ylabel("MSE of W")
plt.title("Plot of MSE of W w.r.t log epsilon")
plt.show()
plt.plot(list_epsilon_power, list_elapsed_time, "bo")
plt.plot([-3, -7], [closed_running_time, closed_running_time])
plt.xlabel("log epsilon")
plt.ylabel("Running time")
plt.title("Plot of running time w.r.t log epsilon")
plt.legend(["Gradient descent validation running time",'Closed form running time'])
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

    W, Q1, Q2 = lr.gradient_descent_linear_regression_alg_old(X_training_set_Adv, Y_training_set_Adv, 10**epsilon_power, 10**eta_power, 10**T_Robbins_Monroe_pow)


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
plt.xlabel("log eta")
plt.ylabel("MSE")
plt.title("Plot of validation MSE w.r.t log eta")
plt.legend(["Gradient descent validation MSE",'Closed form MSE'])
plt.show()
plt.plot(list_eta_power, list_W_MSE, "bo")
plt.title("Plot of MSE of W w.r.t log eta")
plt.xlabel("log eta")
plt.ylabel("MSE of W")
plt.show()
plt.plot(list_eta_power, list_elapsed_time, "bo")
plt.plot([0, -3], [closed_running_time, closed_running_time])
plt.xlabel("log eta")
plt.ylabel("Running time")
plt.title("Plot of running time w.r.t log eta")
plt.legend(["Gradient descent validation running time",'Closed form running time'])
plt.show()

epsilon_power = -5
eta_power = 0
T_Robbins_Monroe_pow = 2

start_time = time.time()
W, Q1, Q2 = lr.gradient_descent_linear_regression_alg_old(X_training_set_Adv, Y_training_set_Adv, 10**epsilon_power, 10**eta_power, 10**T_Robbins_Monroe_pow)
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

    W, Q1, Q2 = lr.gradient_descent_linear_regression_alg(X_training_set_Adv, Y_training_set_Adv, 10**epsilon_power, 10**eta_power, distance_rate_power, 10**T_Robbins_Monroe_pow)


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
plt.xlabel("log distance rate")
plt.ylabel("MSE")
plt.title("Plot of validation MSE w.r.t log DRP")
plt.legend(["New gradient descent validation MSE",'Closed form MSE', 'Gradient descent MSE'])
plt.show()
plt.plot(list_distance_rate_power, list_W_MSE, "bo")
plt.plot([2, 7],[old_W_MSE, old_W_MSE])
plt.xlabel("log distance rate")
plt.ylabel("MSE of W")
plt.legend(["New gradient descent MSE of W", 'Gradient descent MSE of W'])
plt.title("Plot of MSE of W w.r.t log eta")
plt.show()
plt.plot(list_distance_rate_power, list_elapsed_time, "bo")
plt.plot([2, 7],[old_elapsed_time, old_elapsed_time])
plt.plot([2, 7], [closed_running_time, closed_running_time])
plt.xlabel("log distance rate")
plt.ylabel("Running time")
plt.title("Plot of running time w.r.t log log DRP")
plt.legend(["New gradient descent validation running time", 'Gradient descent running time', 'Closed form running time'])
plt.show()

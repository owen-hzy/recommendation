#!/usr/bin/python

import numpy as np
from numpy.linalg import inv
import pandas as pd
import time
import multiprocessing as mp
import sys

# Define an output queue
output = mp.Queue()

factor = [3]
alpha = 20
lambda_reg = 0.02
R = np.loadtxt("data/R.csv", delimiter=",")

# P is the preference matrix
P = np.zeros((R.shape[0], R.shape[1]))
P[R > 0] = 1

# C is the confidence matrix
C = np.ones((R.shape[0], R.shape[1]))
C += alpha * R

# identity matrix
I_m = np.zeros((R.shape[0], R.shape[0]))
I_n = np.zeros((R.shape[1], R.shape[1]))
np.fill_diagonal(I_m, 1)
np.fill_diagonal(I_n, 1)


def compute_X(X, Y, I_f):
    Y_T_times_Y = np.dot(Y.T, Y)

    for row_index in xrange(R.shape[0]):
        C_u = np.zeros((R.shape[1], R.shape[1]))
        for index in xrange(R.shape[1]):
            C_u[index][index] = C[row_index][index]

        first_part = inv(Y_T_times_Y + np.dot(np.dot(Y.T, C_u - I_n), Y) + lambda_reg * I_f)
        second_part = np.dot(np.dot(Y.T, C_u), P[row_index])

        X[row_index] = np.dot(first_part, second_part)

def compute_X_PARA(Y_T_times_Y, I_f, start_row, end_row, queue):
    # Y_T_times_Y = np.dot(Y.T, Y)

    for row_index in xrange(start_row, end_row):
        C_u = np.zeros((R.shape[1], R.shape[1]))
        for index in xrange(R.shape[1]):
            C_u[index][index] = C[row_index][index]

        first_part = inv(Y_T_times_Y + np.dot(np.dot(Y.T, C_u - I_n), Y) + lambda_reg * I_f)
        second_part = np.dot(np.dot(Y.T, C_u), P[row_index])

        X[row_index] = np.dot(first_part, second_part)

def compute_Y(X, Y, I_f):
    X_T_times_X = np.dot(X.T, X)

    for column_index in xrange(R.shape[1]):
        C_i = np.zeros((R.shape[0], R.shape[0]))
        for index in xrange(R.shape[0]):
            C_i[index][index] = C[index][column_index]

        first_part = inv(X_T_times_X + np.dot(np.dot(X.T, C_i - I_m), X) + lambda_reg * I_f)
        second_part = np.dot(np.dot(X.T, C_i), P[:, column_index])

        Y[column_index] = np.dot(first_part, second_part)

fo_run_status = open("run_status.txt", "a")

for f in factor:
    I_f = np.zeros((f, f))
    np.fill_diagonal(I_f, 1)

    # X = np.loadtxt("data/X_" + str(f) + ".csv", delimiter=",")
    # Y = np.loadtxt("data/Y_" + str(f) + ".csv", delimiter=",")
    X = np.random.uniform(0, 1, size=(R.shape[0], f))
    Y = np.random.uniform(0, 1, size=(R.shape[1], f))

    fo_run_status.write("\n\nfactor: %s\n\n" % (f))

    for i in xrange(1):
        start_time = time.time()
        X_old = np.copy(X)
        Y_old = np.copy(Y)

        compute_Y(X, Y, I_f)
        compute_X(X, Y, I_f)
        # Y_T_times_Y = np.dot(Y.T, Y)
        # single = R.shape[0] / 4
        # load = [[0, single], [single, 2 * single], [2 * single, 3 * single], [3 * single, R.shape[0]]]
        # processes  = [mp.Process(target=compute_X_PARA, args = (X, Y_T_times_Y, I_f, load[i][0], load[i][1])) for i in range(4)]

        X_diff_sum = np.sum(np.power(X - X_old, 2))
        Y_diff_sum = np.sum(np.power(Y - Y_old, 2))

        fo_run_status.write("iteration %d, X_diff_sum: %s, Y_diff_sum: %s; period: %s\n" % (i, X_diff_sum, Y_diff_sum, time.time() - start_time))


    np.savetxt("data/X_" + str(f) + ".csv", X, delimiter=",")
    np.savetxt("data/Y_" + str(f) + ".csv", Y, delimiter=",")

fo_run_status.close()
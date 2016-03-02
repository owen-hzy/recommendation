import numpy as np
import pandas as pd
from numpy.linalg import inv
import time
import os
import constants

# latent factors
# check if previously computed result exist, if so, load latent factors from it
def load_latent_factor(filename, delimiter, shape):
    if (os.path.isfile(filename)):
        return np.loadtxt(filename, delimiter=delimiter)
    else:
        return np.random.uniform(0, 1, size=(shape[0], shape[1]))

X = load_latent_factor("data/X.csv", delimiter=",", shape=(constants.R.shape[0], constants.f))
Y = load_latent_factor("data/Y.csv", delimiter=",", shape=(constants.R.shape[1], constants.f))


# Fix Y to compute X
def compute_X():
    Y_T_times_Y = np.dot(Y.T, Y)

    for row_index in xrange(constants.R.shape[0]):
        # construct the diagonal matrix, n * n matrix
        C_u = np.zeros((constants.R.shape[1], constants.R.shape[1]))
        # fill the diagonal element c^u_ii = c_ui where c_ui is from confidence matrix
        for index in xrange(constants.R.shape[1]):
            C_u[index][index] = constants.C[row_index][index]

        # first part here is (Y.T * C_u * Y + lamda_reg * I_n)^-1
        first_part = inv(Y_T_times_Y + np.dot(np.dot(Y.T, C_u - constants.I_n), Y) + constants.lambda_reg * constants.I_f)

        # second part is Y.T * C_u * p(u)
        second_part = np.dot(np.dot(Y.T, C_u), constants.P[row_index])

        X[row_index] = np.dot(first_part, second_part)

# Fix X to compute Y
def compute_Y():
    X_T_times_X = np.dot(X.T, X)

    for column_index in xrange(constants.R.shape[1]):
        C_i = np.zeros((constants.R.shape[0], constants.R.shape[0]))
        for index in xrange(constants.R.shape[0]):
            C_i[index][index] = constants.C[index][column_index]

        first_part = inv(X_T_times_X + np.dot(np.dot(X.T, C_i - constants.I_m), X) + constants.lambda_reg * constants.I_f)
        second_part = np.dot(np.dot(X.T, C_i), constants.P[:, column_index])

        Y[column_index] = np.dot(first_part, second_part)

#compute 20 times
for i in xrange(8):
    start_time = time.time()
    # make a copy, but not a reference
    X_old = np.copy(X)
    Y_old = np.copy(Y)

    # Compute Y first, then compute X, to use to formulation for explanation
    compute_Y()
    compute_X()

    # Check if the computation is stabilized
    X_diff_sum = np.sum(np.power(X - X_old, 2))
    Y_diff_sum = np.sum(np.power(Y - Y_old, 2))

    print "X_diff_sum: %s   ;   Y_diff_sum: %s" % (X_diff_sum, Y_diff_sum)
    print "---- %s seconds ----" % (time.time() - start_time)

# save the computed result to file, to be used in next iteration
np.savetxt("data/X.csv", X, delimiter=",")
np.savetxt("data/Y.csv", Y, delimiter=",")


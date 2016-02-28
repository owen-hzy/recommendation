import numpy as np
import pandas as pd
from numpy.linalg import inv
import time

### global parameter, parameter to be tuned later on
alpha = 40
lambda_reg = 2
f = 3

train_data = pd.read_csv("data/train_data.csv")   # train data only contains records with positive quantity

#change train from data frame to matrix form, the name is R to be consistent with paper, where R is the record matrix with row being user,
# column being item and the value being the variable to be observed
R = pd.pivot_table(train_data, values="quantity", index=["customer_id"], columns=["item_id"], aggfunc=np.sum, fill_value=0).values

P = np.zeros((R.shape[0], R.shape[1]))
P[R > 0] = 1

# C is the confidence matrix
C = np.ones((R.shape[0], R.shape[1]))
C += alpha * R  # c_ui = 1 + alpha * r_ui

# latent factors
X = np.random.uniform(0, 1, size=(R.shape[0], f))
Y = np.random.uniform(0, 1, size=(R.shape[1], f))

# identity matrix with dimension same as dimension of latent factor
I_f = np.zeros((f, f))
np.fill_diagonal(I_f, 1)

# identity matrix with dimension m * m and n * n, where m is the number of user, n is the number of item
I_m = np.zeros((R.shape[0], R.shape[0]))
I_n = np.zeros((R.shape[1], R.shape[1]))
np.fill_diagonal(I_m, 1)
np.fill_diagonal(I_n, 1)

# Fix Y to compute X
def compute_X():
    Y_T_times_Y = np.dot(Y.T, Y)

    for row_index in xrange(R.shape[0]):
        # construct the diagonal matrix, n * n matrix
        C_u = np.zeros((R.shape[1], R.shape[1]))
        # fill the diagonal element c^u_ii = c_ui where c_ui is from confidence matrix
        for index in xrange(R.shape[1]):
            C_u[index][index] = C[row_index][index]

        # first part here is (Y.T * C_u * Y + lamda_reg * I_n)^-1
        first_part = inv(Y_T_times_Y + np.dot(np.dot(Y.T, C_u - I_n), Y) + lambda_reg * I_f)

        # second part is Y.T * C_u * p(u)
        second_part = np.dot(np.dot(Y.T, C_u), P[row_index])

        X[row_index] = np.dot(first_part, second_part)

# Fix X to compute Y
def compute_Y():
    X_T_times_X = np.dot(X.T, X)

    for column_index in xrange(R.shape[1]):
        C_i = np.zeros((R.shape[0], R.shape[0]))
        for index in xrange(R.shape[0]):
            C_i[index][index] = C[index][column_index]

        first_part = inv(X_T_times_X + np.dot(np.dot(X.T, C_i - I_m), X) + lambda_reg * I_f)
        second_part = np.dot(np.dot(X.T, C_i), P[:, column_index])

        Y[column_index] = np.dot(first_part, second_part)


# compute 20 times
for i in xrange(2):
    start_time = time.time()
    # make a copy, but not a reference
    X_old = np.copy(X)
    Y_old = np.copy(Y)

    compute_X()
    compute_Y()

    X_diff_sum = np.sum(np.power(X - X_old, 2))
    Y_diff_sum = np.sum(np.power(Y - Y_old, 2))

    print "X_diff_sum: %s   ;   Y_diff_sum: %s" % (X_diff_sum, Y_diff_sum)
    print "---- %s seconds ----" % (time.time() - start_time)

# save the computed result to file
np.savetxt("data/X.csv", X, delimiter=",")
np.savetxt("data/Y.csv", Y, delimiter=",")


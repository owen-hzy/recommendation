import numpy as np
from numpy.linalg import inv
import os
import constants

# load the latent factor from computed result
X = np.loadtxt("data/X.csv", delimiter=",")
Y = np.loadtxt("data/Y.csv", delimiter=",")

# P_hat the predicted matrix
P_hat = np.dot(X, Y.T)

# get max 3 recommended item index for user u
def max_index_from_predicted_matrix_of_specific_user(u, number_of_item=3):
    # sort ascending by row(axis = 1)
    item_index = np.argsort(P_hat[u])
    item_index_size = np.size(item_index)
    return item_index[item_index_size : item_index_size - (number_of_item + 1): -1]

# max 3 item for 10th user
max_predicted_item_index = max_index_from_predicted_matrix_of_specific_user(9)

# check the largest first
i = max_predicted_item_index[0]

# np.nonzero returns tuple
P_u_nonzero_index = np.nonzero(constants.P[9])[0]

def get_C_u(u):
    C_u = np.zeros((constants.R.shape[1], constants.R.shape[1]))
    # fill the diagonal element c^u_ii = c_ui where c_ui is from confidence matrix
    for index in xrange(constants.R.shape[1]):
        C_u[index][index] = constants.C[u][index]
    return C_u

C_u = get_C_u(9)

W_u = inv(np.dot(Y.T, Y) + np.dot(np.dot(Y.T, C_u - constants.I_n), Y) + constants.lambda_reg * constants.I_f)
s_u = np.dot(np.dot(Y[i], W_u), Y.T[:, P_u_nonzero_index])
c_u = constants.C[9][P_u_nonzero_index]

p_hat_ui_decomposed_array = np.multiply(s_u, c_u)
decomposed_array_index = np.argsort(p_hat_ui_decomposed_array)[::-1]

# purchased item id array sorted from contributing most to least
P_u_nonzero_index[decomposed_array_index]
# predicted preference array sorted from contributing most to least
p_hat_ui_decomposed_array[decomposed_array_index]




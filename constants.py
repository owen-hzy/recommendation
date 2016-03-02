import pandas as pd
import numpy as np

### global parameter, parameter to be tuned later on
alpha = 40
lambda_reg = 2
f = 3

train_data = pd.read_csv("data/train_data.csv")   # train data only contains records with positive quantity
test_data = pd.read_csv("data/test_data.csv")

#change train from data frame to matrix form, the name is R to be consistent with paper, where R is the record matrix with row being user,
# column being item and the value being the variable to be observed
train_matrix = pd.pivot_table(train_data, values="quantity", index=["customer_id"], columns=["item_id"], aggfunc=np.sum, fill_value=0)
test_matrix = pd.pivot_table(test_data, values="quantity", index=["customer_id"], columns=["item_id"], aggfunc=np.sum, fill_value=0)

R = train_matrix.values
R_t = test_matrix.Cvalues

P = np.zeros((R.shape[0], R.shape[1]))
P[R > 0] = 1

# C is the confidence matrix
C = np.ones((R.shape[0], R.shape[1]))
C += alpha * R  # c_ui = 1 + alpha * r_ui

# identity matrix with dimension same as dimension of latent factor
I_f = np.zeros((f, f))
np.fill_diagonal(I_f, 1)

# identity matrix with dimension m * m and n * n, where m is the number of user, n is the number of item
I_m = np.zeros((R.shape[0], R.shape[0]))
I_n = np.zeros((R.shape[1], R.shape[1]))
np.fill_diagonal(I_m, 1)
np.fill_diagonal(I_n, 1)
import numpy as np
import pandas as pd
import sys
from sklearn.metrics.pairwise import pairwise_distances

# find out all the non-zero element index in R_t, where column is customer_id, and column is item_id, value is quantity
test_data = pd.read_csv("data/test_data.csv")
train_data = pd.read_csv("data/train_data.csv")

test_matrix = pd.pivot_table(test_data, values="quantity", index=["customer_id"], columns=["item_id"], aggfunc=np.sum, fill_value=0)
train_matrix = pd.pivot_table(train_data, values="quantity", index=["customer_id"], columns=["item_id"], aggfunc=np.sum, fill_value=0)
#train_matrix = pd.read_csv("data/train_data_item.csv", index_col=0)


R_t = test_matrix.values
test_nonzero_index = np.nonzero(R_t)

# locate the non-zero customer and item in train matrix by the customer id and item id
customer_id_index_in_train = []
item_id_index_in_train = []
for i in xrange(len(test_nonzero_index[0])):
    customer_id_index_in_train.append(np.where(train_matrix.index == test_matrix.index[test_nonzero_index[0][i]])[0][0])
    item_id_index_in_train.append(np.where(train_matrix.columns == test_matrix.columns[test_nonzero_index[1][i]])[0][0])

print len(customer_id_index_in_train)
print len(item_id_index_in_train)
sys.stdout.flush()
for f in [3]:

    # get the predicted matrix
    X = np.loadtxt("data/X_" + str(f) + ".csv", delimiter=",")
    Y = np.loadtxt("data/Y_" + str(f) + ".csv", delimiter=",")
    P_hat = np.dot(X, Y.T)
    # arg sort by predicted preference
    P_hat_sort = np.argsort(P_hat)
#
#     ############# Popularity based model
#     # recommend based on the popularity, e.g. the one purchased the most be recommended first
#     # popularity = np.sum(constants.train_matrix, axis=0)
#     # popularity_rank = np.argsort(popularity)[::-1]
#
#     ############# Item-based neighborhood model
#     # S_neighbor = np.subtract(1, pairwise_distances(constants.R.T, metric="cosine"))
#     # P_neighbor = np.dot(constants.R, S_neighbor)
#     # P_neighbor_sort = np.argsort(P_neighbor)
#
    rank_ui = []
    # rank_ui_popularity = []
    # rank_ui_neighbor = []
    r_t_ui = []
    for i in xrange(len(customer_id_index_in_train)):
        rank_ui.append(np.where(P_hat_sort[customer_id_index_in_train[i]][::-1] == item_id_index_in_train[i])[0][0])
        # rank_ui_popularity.append(np.where(popularity_rank == item_id_index_in_train[i])[0][0])
        # rank_ui_neighbor.append(np.where(P_neighbor_sort[customer_id_index_in_train[i]][::-1] == item_id_index_in_train[i])[0][0])
        r_t_ui.append(R_t[test_nonzero_index[0][i]][test_nonzero_index[1][i]])

    # number of item
    n = P_hat.shape[1]
#
    rank_ui = np.true_divide(rank_ui, n)
    # rank_ui_popularity = np.true_divide(rank_ui_popularity, n)
    # rank_ui_neighbor = np.true_divide(rank_ui_neighbor, n)
    rank_bar = np.average(rank_ui, weights=r_t_ui)
    # rank_bar_popularity = np.average(rank_ui_popularity, weights=r_t_ui)
    # rank_bar_neighbor = np.average(rank_ui_neighbor, weights=r_t_ui)

    print "factor: %s, rank_bar: %s" % (f, rank_bar)

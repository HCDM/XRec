import numpy as np
import math
# import decision_tree as dt
# import optimization as opt
from tree import *
import argparse
from sklearn.metrics import mean_squared_error
import os

def getRatingMatrix(filename):
    data = []
    data_fo = []
    feature = []
    with open(filename) as file:
        for line in file:
            d = line[:-1].split(",")
            list1 = [int(x) for x in d[:-1]]
            list2 = [int(x) for x in d[-1].split(" ")]

            data.append(list1)
            data_fo.append(list2)
            for i in list2:
                feature.append(i)
    data = np.array(data)

    num_users = data[:, 0].max() + 1
    num_items = data[:, 1].max() + 1
    num_features = max(feature) + 1
    print num_features
    
    # create rating matrix, and user_opinion, item_opinion matrices
    # user_opinion: user preference for each feature
    # item_opinion: item performance on each feature
    rating_matrix = np.zeros((num_users, num_items), dtype=float)
    user_opinion = np.full((num_users, num_features), np.nan)
    item_opinion = np.full((num_items, num_features), np.nan)
    # update the matrices with input data
    # get the accumulated feature opinion scores for users and items.
    for i in range(len(data)):
        user_id, item_id, rating = data[i]
        rating_matrix[user_id][item_id] = rating
        num_pos = 0
        num_neg = 0
        for j in range(0, len(data_fo[i]), 2):
            # for user, count the frequency
            if np.isnan(user_opinion[user_id][data_fo[i, j]]):
                user_opinion[user_id][data_fo[i, j]] = 1
            else:
                user_opinion[user_id][data_fo[i, j]] += 1
            # for item, count the sentiment score
            if np.isnan(item_opinion[item_id][data_fo[i, j]]):
                item_opinion[item_id][data_fo[i, j]] = data_fo[i, j+1]
            else:
                item_opinion[item_id][data_fo[i, j]] += data_fo[i, j+1]

    return rating_matrix, user_opinion, item_opinion

def dcg_k(r, k):
    r = np.asfarray(r)[:k]
    if r.size != k:
        raise ValueError('ranking list length < k')
    return np.sum(r / np.log2(np.arange(2, r.size + 2)))

def ndcg_k(r, k):
    sorted_r = sorted(r, reverse=True)
    idcg = dcg_k(sorted_r, k)
    if not idcg:
        return 0
    return dcg_k(r, k) / idcg

def get_ndcg(prediction, rating_matrix, k):

    num_user, num_item = rating_matrix.shape
    ndcg_overall = []
    for i in range(num_user):
        # skip the user without any rating
        if rating_matrix[i].sum() == 0:
            continue
        else:
            pred_list_index = np.argsort(-prediction[i])
            pred_true_rating = rating_matrix[i][pred_list_index]
            ndcg_overall.append(ndcg_k(pred_true_rating, k))
    return np.mean(ndcg_overall)

def MatrixFactorization(num_dim, lr, lambda_u, lambda_v, num_iters, rating_matrix):

    # get the nonzero values of the rating matrix
    np.random.seed(0)
    user_index, item_index = rating_matrix.nonzero()
    mask_matrix = rating_matrix.nonzero()
    num_records = len(user_index)
    num_users = user_index.max() + 1
    num_items = item_index.max() + 1
    # randomly initialize the user, item vectors.
    user_vector = np.random.rand(num_users, num_dim)
    item_vector = np.random.rand(num_items, num_dim)

    for it in range(num_iters):
        for i in range(num_records):
            u_id, v_id = user_index[i], item_index[i]
            r = rating_matrix[u_id, v_id]
            # update latent factors of users and items
            user_vector[u_id] += lr * ((r - np.dot(user_vector[u_id], item_vector[v_id])) * item_vector[v_id] - lambda_u * user_vector[u_id])
            item_vector[v_id] += lr * ((r - np.dot(user_vector[u_id], item_vector[v_id])) * user_vector[u_id] - lambda_v * item_vector[v_id])
            
        # calculte the training error
        pred = np.dot(user_vector, item_vector.T)
        error = mean_squared_error(pred[mask_matrix], rating_matrix[mask_matrix])
    return user_vector, item_vector

def AlternativeOptimization(rating_matrix, user_opinion, item_opinion, num_dim, max_depth, num_BPRpairs, lr, lambda_u, lambda_v,
                            lambda_BPR, num_run, num_iter_user, num_iter_item, batch_size, random_seed):
    num_users, num_items = rating_matrix.shape
    num_features = user_opinion.shape[1]
    print "Number of users", num_users 
    print "Number of items", num_items
    print "Number of features", num_features
    print "Number of latent dimensions: ", num_dim
    print "Maximum depth of the regression tree: ", max_depth

    user_vector, item_vector = MatrixFactorization(num_dim, lr, lambda_u, lambda_v, 50, rating_matrix)
    pred = np.dot(user_vector, item_vector.T)

    i = 0
    while i < num_run:
        user_vector_old = user_vector
        iterm_vector_old = item_vector
        pred_old = pred

        print "********** Round", i, "create user tree **********"
        user_tree = Tree(Node(None, 1), rating_matrix=rating_matrix, opinion_matrix=user_opinion, anchor_vectors=item_vector, lr=lr,
                         num_dim=num_dim, max_depth=max_depth, num_BPRpairs=num_BPRpairs, lambda_anchor=lambda_v, lambda_target=lambda_u, 
                         lambda_BPR=lambda_BPR, num_iter=num_iter_user, batch_size=batch_size, random_seed=random_seed)
        # create the user tree with the known item latent factors
        user_tree.create_tree(user_tree.root, user_tree.opinion_matrix, user_tree.rating_matrix)
        print "get user vectors"
        user_vector = user_tree.get_vectors()
        # add the refinement to the leave nodes of user tree as personalized representation
        print "add personalized term"
        user_vector = user_tree.personalization(user_vector)

        print "********** Round", i, "create item tree **********"
        item_tree = Tree(Node(None, 1), rating_matrix=rating_matrix.T, opinion_matrix=item_opinion, anchor_vectors=user_vector, lr=lr,
                        num_dim=num_dim, max_depth=max_depth, num_BPRpairs=num_BPRpairs, lambda_anchor=lambda_u, lambda_target=lambda_v,
                        lambda_BPR=lambda_BPR, num_iter=num_iter_item, batch_size=batch_size, random_seed=random_seed)
        # create the item tree with the learned user latent factors
        item_tree.create_tree(item_tree.root, item_tree.opinion_matrix, item_tree.rating_matrix)
        item_vector = item_tree.get_vectors()
        # add the refinement to the leave nodes of item tree as personalized representation
        item_vector = item_tree.personalization(item_vector)

        pred = np.dot(user_vector, item_vector.T)
        error = LA.norm(pred_old - pred) ** 2
        if error < 0.1:
            break
        i = i + 1
    return user_tree, item_tree, user_vector, item_vector


if __name__ == "__main__":
    # initialization
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", help="training filename", default="../data/yelp_train.txt")
    parser.add_argument("--test_file", help="test filename", default="../data/yelp_test.txt")
    parser.add_argument("--num_dim", help="the number of latent dimension", default=20)
    parser.add_argument("--max_depth", help="the maximum depth of the tree", default=6)
    parser.add_argument("--lambda_u", help="regularization parameter for user vectors", default=1)
    parser.add_argument("--lambda_v", help="regularization parameter for item vectors", default=1)
    parser.add_argument("--lambda_bpr", help="regularization parameter for BPR term", default=100)
    parser.add_argument("--num_BPRpairs", help="number of BPR pairs for each udpate", default=20)
    parser.add_argument("--batch_size", help="batch size for stochastic gradient descent", default=100)
    parser.add_argument("--num_iter_user", help="number of iterations for user vector update", default=20)
    parser.add_argument("--num_iter_item", help="number of iterations for item vector udpate", default=1000)
    parser.add_argument("--learning_rate", help="learning rate for sgd", default=0.05)
    parser.add_argument("--random_seed", help="random seed for initialization and BPR calculation", default=0)
    parser.add_argument("--num_run", help="number of iterations for alternatively creating the trees", default=5)

    args = parser.parse_args()
    train_file = args.train_file
    test_file = args.test_file
    NUM_DIM = int(args.num_dim)
    MAX_DEPTH = int(args.max_depth)
    LAMBDA_U = float(args.lambda_u)
    LAMBDA_V = float(args.lambda_v)
    LAMBDA_BPR = float(args.lambda_bpr)
    NUM_BPRPAIRS = int(args.num_BPRpairs)
    BATCH_SIZE = int(args.batch_size)
    NUM_ITER_U = int(args.num_iter_user)
    NUM_ITER_V = int(args.num_iter_item)
    lr = float(args.learning_rate)
    random_seed = int(args.random_seed)
    NUM_RUN = int(args.num_run)
    print "********** Load training data **********"
    rating_matrix, user_opinion, item_opinion = getRatingMatrix(train_file)

    # build the factorization tree with the training dataset
    user_tree, item_tree, user_vector, item_vector = AlternativeOptimization(rating_matrix=rating_matrix,
                                                                           user_opinion=user_opinion, item_opinion=item_opinion,
                                                                           num_dim=NUM_DIM, max_depth=MAX_DEPTH,
                                                                           num_BPRpairs=NUM_BPRPAIRS, lr=lr,
                                                                           lambda_u=LAMBDA_U, lambda_v=LAMBDA_V,
                                                                           lambda_BPR=LAMBDA_BPR, num_run=NUM_RUN,
                                                                           num_iter_user=NUM_ITER_U, num_iter_item=NUM_ITER_V,
                                                                           batch_size=BATCH_SIZE, random_seed=random_seed)
    pred_rating = np.dot(user_vector, item_vector.T)
    # save the results
    if not os.path.exists("../results/"):
        os.makedirs(directory)
    np.savetxt("../results/item_vector.txt", item_vector, fmt='%0.8f')
    np.savetxt("../results/user_vector.txt", user_vector, fmt="%0.8f")
    np.savetxt("../results/pred_rating.txt", pred_rating, fmt="%0.8f")

    # test on test data with the trained model
    print "********** Load test data **********"
    test_rating, user_opinion_test, item_opinion_test = getRatingMatrix(test_file)
    print "Number of users", test_rating.shape[0]
    print "Number of items", test_rating.shape[1]
    print "Number of features", user_opinion.shape[1]

    # get the NDCG results
    print "********** User tree **********"
    user_tree.print_tree(user_tree.root)
    print "********** Item tree **********" 
    item_tree.print_tree(item_tree.root)
    print "********** NDCG **********"
    ndcg_10 = get_ndcg(pred_rating, test_rating, 10)
    print "NDCG@10: ", ndcg_10
    ndcg_20 = get_ndcg(pred_rating, test_rating, 20)
    print "NDCG@20: ", ndcg_20
    ndcg_50 = get_ndcg(pred_rating, test_rating, 50)
    print "NDCG@50: ", ndcg_50

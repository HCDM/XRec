# import optimization as opt
import multiprocessing as mp
import numpy as np
from numpy import linalg as LA
from copy_reg import pickle
from types import MethodType


class Node:
    def __init__(self, parent_node, node_depth):
        self.feature_index = None
        self.predicate = 0
        self.parent = parent_node
        self.depth = node_depth
        self.left = None
        self.right = None
        self.empty = None
        self.vector = None

class Tree:
    def __init__(self, root_node, rating_matrix, opinion_matrix, anchor_vectors, lr, num_dim, max_depth, num_BPRpairs, lambda_anchor, lambda_target, lambda_BPR, num_iter, batch_size, random_seed):
        self.random_seed = random_seed
        np.random.seed(self.random_seed)
        self.root = root_node
        self.root.vector = np.random.rand(num_dim)
        self.rating_matrix = rating_matrix
        self.num_target, self.num_anchor = rating_matrix.shape
        self.opinion_matrix = opinion_matrix
        self.num_feature = opinion_matrix.shape[1]
        self.num_dim = num_dim
        self.max_depth = max_depth
        self.num_BPRpairs = num_BPRpairs
        self.lambda_anchor = lambda_anchor
        self.lambda_target = lambda_target
        self.lambda_BPR = lambda_BPR
        self.anchor_vectors = anchor_vectors
        self.mask_matrix = np.nonzero(rating_matrix)
        self.num_iter = num_iter
        self.batch_size = batch_size
        self.lr = lr
        self.num_bin = 5

    def personalization(self, vector):
        for i in range(self.num_target):
            user_rating = self.rating_matrix[np.array([i])]
            vector[i] = self.sgd_update(user_rating, vector[i])
        return vector

    def calculate_loss(self, current_node, rating_matrix):
        np.random.seed(self.random_seed)
        target_vectors = np.array([current_node.vector for i in range(rating_matrix.shape[0])])
        value = 0

        # add l2 difference term
        mask_matrix = np.nonzero(rating_matrix)
        pred = np.dot(target_vectors, self.anchor_vectors.T)
        value += LA.norm((rating_matrix - pred)[mask_matrix]) ** 2

        # add regularization term
        target_reg = LA.norm(target_vectors) ** 2
        anchor_reg = LA.norm(self.anchor_vectors) ** 2
        value += self.lambda_anchor * anchor_reg + self.lambda_target * target_reg
        
        # add BPR term
        value += self.get_BPR(rating_matrix, target_vectors)

        return value

    def get_predicate(self, opinion_matrix, feature_index):
        op = opinion_matrix[:, feature_index]
        sorted_op = sorted(op)
        num_known = np.count_nonzero(~np.isnan(sorted_op))
        bin_interval = num_known // self.num_bin
        predicate_list = np.unique([sorted_op[(i + 1) * bin_interval] for i in range(self.num_bin - 1)])
        return predicate_list

    def split(self, opinion_matrix, feature_index, split_value):
        # divide the users/items into three partitions according to the feature opinion
        index_left = np.where(opinion_matrix[:, feature_index] >= split_value)[0]
        index_right = np.where(opinion_matrix[:, feature_index] < split_value)[0]
        index_empty = np.where(opinion_matrix[:, feature_index] == np.nan)[0]
        
        return index_left, index_right, index_empty

    def sgd(self, rating_matrix, target_vector):
        delta = np.zeros_like(target_vector)
        num_t, num_a = rating_matrix.shape
        if num_t == 0:
            return delta

        np.random.seed(self.random_seed)
        for i in range(self.batch_size):
            idx = np.random.randint(0, num_t * num_a)
            t, a = idx // num_a, idx % num_a
            if rating_matrix[t, a] != 0:
                delta += -2 * (rating_matrix[t, a] - np.dot(target_vector, self.anchor_vectors[a])) * self.anchor_vectors[a] + 2 * self.lambda_target * target_vector

        for i in range(self.num_BPRpairs):
            id1, id2 = np.random.randint(0, num_t * num_a, 2)
            t1, a1 = id1 // num_a, id1 % num_a
            t2, a2 = id2 // num_a, id2 % num_a
            if rating_matrix[t1, a1] > rating_matrix[t2, a2]:
                diff = np.dot(target_vector.T, self.anchor_vectors[a1] - self.anchor_vectors[a2])
                diff = -diff
                delta += self.lambda_BPR * (self.anchor_vectors[a2] - self.anchor_vectors[a1]) * np.exp(diff) / (1 + np.exp(diff))
        
        return delta

    def sgd_update(self, index_matrix, current_vector):
        if len(index_matrix) <= 0:
            return current_vector
        np.random.seed(self.random_seed)
        target_vector = np.random.random(size=current_vector.shape)

        eps = 1e-8
        sum_square = eps + np.zeros_like(target_vector)
        # SGD procedure
        for i in range(self.num_iter):
            delta = self.sgd(index_matrix, current_vector + target_vector)
            sum_square += np.square(delta)
            lr_t = np.divide(self.lr, np.sqrt(sum_square))
            target_vector -= lr_t * delta
        # induce the latent factors from parent nodes
        target_vector += current_vector
        
        return target_vector
            
    def calculate_subtree_value(self, index_matrix, current_vector):
        mmatrix = np.nonzero(index_matrix)
        vector = self.sgd_update(index_matrix, current_vector)
        vector = np.array([vector for i in range(index_matrix.shape[0])])
        pred = np.dot(vector, self.anchor_vectors.T)
        err = LA.norm((pred - index_matrix)[mmatrix]) ** 2

        return err, vector

    def calculate_splitvalue(self, rating_matrix, current_vector, index_left, index_right, index_empty):
        # calculate the loss of the partition
        left = rating_matrix[index_left]
        right = rating_matrix[index_right]
        empty = rating_matrix[index_empty]
        left_vector = np.zeros(self.num_dim)
        right_vector = np.zeros(self.num_dim)
        empty_vector = np.zeros(self.num_dim) 
        value = 0

        if len(index_left) > 0:
            err, left_vector = self.calculate_subtree_value(left, current_vector)
            value += err
        if len(index_right) > 0:
            err, right_vector = self.calculate_subtree_value(right, current_vector)
            value += err
        if len(index_empty) > 0:
            err, empty_vector = self.calculate_subtree_value(empty, current_vector)
            value += err

        value += self.lambda_target * (LA.norm(left_vector) ** 2 + LA.norm(right_vector) ** 2 + LA.norm(empty_vector) ** 2)
        value += self.lambda_anchor * (LA.norm(self.anchor_vectors) ** 2)
        value += self.get_BPR(left, left_vector)
        value += self.get_BPR(right, right_vector)
        value += self.get_BPR(empty, empty_vector)
        
        return value

    def get_BPR(self, rating_matrix, target_vector):
        np.random.seed(self.random_seed)
        num_t, num_a = rating_matrix.shape
        if num_t * num_a == 0:
            return 0 
        value = 0
        # randomly sample num_BPRpairs pairs to get the BPR loss
        for i in range(self.num_BPRpairs):
            p1, p2 = np.random.randint(0, num_t * num_a, 2)
            t1, a1 = p1 // num_a, p1 % num_a
            t2, a2 = p2 // num_a, p2 % num_a
            if rating_matrix[t1, a1] > rating_matrix[t2, a2]:
                diff = np.dot(target_vector[t1].T, self.anchor_vectors[a1]) - np.dot(target_vector[t2].T, self.anchor_vectors[a2])
                diff = -diff
                value += self.lambda_BPR * np.log(1 + np.exp(diff))
        
        return value

    def print_tree(self, current_node, level=0):
        if current_node == None:
            return
        print '\t' * level, current_node.feature_index
        self.print_tree(current_node.left, level + 1)
        self.print_tree(current_node.right, level + 1)
        self.print_tree(current_node.empty, level + 1)

    def get_vectors(self):
        vectors = np.zeros((self.num_target, self.num_dim))
        for i in range(self.num_target):
            current_node = self.root

            while current_node.left != None or current_node.right != None or current_node.empty != None:
                if np.isnan(self.opinion_matrix[i][current_node.feature_index]):
                    current_node = current_node.empty
                elif self.opinion_matrix[i][current_node.feature_index] < current_node.predicate:
                    current_node = current_node.left
                elif self.opinion_matrix[i][current_node.feature_index] >= current_node.predicate:
                    current_node = current_node.right
            vectors[i] = current_node.vector
        return vectors

    def __call__(self, rating_matrix, current_vector, index_left, index_right, index_empty):
        r = self.calculate_splitvalue(rating_matrix, current_vector, index_left, index_right, index_empty)
        return r

    def get_best_predicate(self, current_node, opinion_matrix, rating_matrix):
        predicate = np.full((self.num_feature, self.num_bin - 1), np.nan)
        split_value = np.full((self.num_feature, self.num_bin - 1), np.inf)

        # get all the predicate
        for feature_index in range(self.num_feature):
            predicate_list = self.get_predicate(opinion_matrix, feature_index)
            predicate[feature_index][:len(predicate_list)] = predicate_list
        # calculate all the predicate values
        for feature_index in range(self.num_feature):
            for predicate_index in range(self.num_bin - 1):
                if np.isnan(predicate[feature_index][predicate_index]):
                    continue
                else:
                    print feature_index, predicate[feature_index][predicate_index]
                    index_left, index_right, index_empty = self.split(opinion_matrix, feature_index, predicate[feature_index][predicate_index])
                    split_value[feature_index][predicate_index] = self.calculate_splitvalue(rating_matrix, current_node.vector, index_left, index_right, index_empty)

        # find the best predicate: feature index, predicate value
        ridx, cidx = np.where(split_value == split_value.min())
        # only use the first if several minimums are returned
        ridx = ridx[0]
        cidx = cidx[0]

        return split_value.min(), ridx, predicate[ridx][cidx]

    def get_best_predicate_parallel(self, current_node, opinion_matrix, rating_matrix):
        predicate = np.full((self.num_feature, self.num_bin - 1), np.nan)
        split_value = np.full((self.num_feature, self.num_bin - 1), np.inf)

        # get all the predicate
        for feature_index in range(self.num_feature):
            predicate_list = self.get_predicate(opinion_matrix, feature_index)
            predicate[feature_index] = predicate_list

        # prepare the parameters for parallel computing
        params = {}
        c = 0
        for feature_index in range(self.num_feature):
            params[feature_index] = {}
            for predicate_index in range(self.num_bin - 1):
                if np.isnan(predicate[feature_index][predicate_index]):
                    continue
                else:
                    params[feature_index][predicate_index] = []
                    index_left, index_right, index_empty = self.split(opinion_matrix, feature_index, predicate[feature_index][predicate_index])
                    params[feature_index][predicate_index].extend(rating_matrix, current_node,vector, index_left, index_right, index_empty)
                    c += 1
        #### set up multiprocessing
        pool = mp.Pool()
        results = {}
        for feature_index in range(self.num_feature):
            results[feature_index] = []
            for predicate_index in range(self.num_bin - 1):
                if np.isnan(predicate[feature_index][predicate_index]):
                    continue
                else:
                    result = pool.apply_async(self, params[feature_index][predicate_index])
                    results[feature_index].append(result)
        for feature_index in range(self.num_feature):
            for predicate_index in range(self.num_bin - 1):
                try:
                    split_value[feature_index][predicate_index] = results[feature_index][predicate_index].get()
                except:
                    continue

        ridx, cidx = np.where(split_value == split_value.min())
        # only use the first if several minimums are returned
        ridx = ridx[0]
        cidx = cidx[0]

        return split_value.min(), ridx, predicate[ridx][cidx]

    def create_tree(self, current_node, opinion_matrix, rating_matrix):
        print "Current depth: ", current_node.depth
        if current_node.depth > self.max_depth:
            print ">>>>>>>>>>>>>>>>>>> STOP: tree depth exceeds the maximum limit."
            return
        if len(rating_matrix) == 0:
            print ">>>>>>>>>>>>>>>>>>> STOP: No rating matrix."
            return

        error_old = self.calculate_loss(current_node, rating_matrix)

        min_split_value, best_feature, best_predicate = self.get_best_predicate(current_node, opinion_matrix, rating_matrix)
        print min_split_value, best_feature, best_predicate
        current_node.feature_index = best_feature
        current_node.predicate = best_predicate
        ### you can also use the parallel version
        # best_feature, best_predicate = self.get_best_predicate_parallel(current_node, opinion_matrix, rating_matrix)

        index_left, index_right, index_empty = self.split(opinion_matrix, best_feature, best_predicate)
        left_rating_matrix, left_opinion_matrix = rating_matrix[index_left], opinion_matrix[index_left]
        right_rating_matrix, right_opinion_matrix = rating_matrix[index_right], opinion_matrix[index_right]
        empty_rating_matrix, empty_opinion_matrix = rating_matrix[index_empty], opinion_matrix[index_empty]

        # get the updated latent representation of each child node
        left_vector = self.sgd_update(left_rating_matrix, current_node.vector)
        right_vector = self.sgd_update(right_rating_matrix, current_node.vector)
        empty_vector = self.sgd_update(empty_rating_matrix, current_node.vector)

        # recursively create the tree for the child node until covernge
        if min_split_value < error_old:
            # left child tree
            current_node.left = Node(parent_node=current_node, node_depth=current_node.depth + 1)
            current_node.left.vector = left_vector
            if len(left_rating_matrix) != 0:
                self.create_tree(current_node.left, left_opinion_matrix, left_rating_matrix)
            # right child tree
            current_node.right = Node(parent_node=current_node, node_depth=current_node.depth + 1)
            current_node.right.vector = right_vector
            if len(right_rating_matrix) != 0:
                self.create_tree(current_node.right, right_opinion_matrix, right_rating_matrix)
            # empty child tree
            current_node.empty = Node(parent_node=current_node, node_depth=current_node.depth + 1)
            current_node.empty.vector = empty_vector
            if len(empty_rating_matrix) != 0:
                self.create_tree(current_node.empty, empty_opinion_matrix, empty_rating_matrix)
        else:
            print ">>>>>>>>>>>>>>>>>>> STOP: cannot not be split any more."
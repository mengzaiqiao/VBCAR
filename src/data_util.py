"""
Created on Aug 5, 2019
Updated on XX,2019 BY xxx@

Classes describing datasets of user-item interactions. Instances of these
are returned by dataset fetching and dataset pre-processing functions.

@author: Zaiqiao Meng (zaiqiao.meng@gmail.com)

"""

import numpy as np
import pandas as pd
import math

pd.options.mode.chained_assignment = None
import os
import sys
sys.path.append("./")
sys.path.append("../")

import numpy as np
import random
from src.unigramTable import UnigramTable
from src.python_evaluation import map_at_k, ndcg_at_k, precision_at_k, recall_at_k

TOP_K = 10

instacart_dir = "./datasets/instacart/data/Instacart_"
tafeng_dir = "./datasets/ta-feng/data/ta-feng_"
dunnhumby_dir = "./dataset/dunnhumby/data/dunnhumby_"

import numpy as np
import pandas as pd
import sklearn
import os

DEFAULT_USER_COL = "user_ids"
DEFAULT_ITEM_COL = "item_ids"
DEFAULT_ORDER_COL = "order_ids"
DEFAULT_RATING_COL = "ratings"
DEFAULT_LABEL_COL = "label"
DEFAULT_TIMESTAMP_COL = "timestamp"
DEFAULT_PREDICTION_COL = "prediction"
DEFAULT_FLAG_COL = "flag"

validate_size = 0.2
test_size = 0.2
negative_size = 100

min_u_c = 16  # Filter items by mixmum number of orders
min_i_c = 30  # Filter users by mixmum number of items
min_o_c = 7  # Filter users by mixmum number of orders


import random
import numpy as np
import pandas as pd
import warnings


class Dataset(object):
    """
    Data Object.
    
    Parameters
    ----------
    train: Dataframe with columns of col_user, col_item, col_rating
    test: Dataframe for testing
    user_fea_dict: Dict of user features
    item_fea_dict: Dict of item features
    n_neg_test: Number of negative samples for testing and validating
    col_user: column name for user
    col_item: column name for item
    col_rating: column name for rating
    col_flag: column name for flag (test, validate, train)
    col_timestamp: column name for timestamp
    seed: random seed
    
    Attributes
    ----------
    user_idx: user_idx
    item_idx: item_idx
    n_users: Number of users
    n_items: Number of items
    n_neg_test: Number of negative samples for testing and validating
    user_fea_dict: user feature dict
    item_fea_dict: tiem feature dict
    col_user: column name for user
    col_item: column name for item
    col_rating: column name for rating
    col_flag: column name for flag (test, validate, train)
    col_timestamp: column name for timestamp
    train: Dataframe of train
    test: Dataframe of test
    validate: Dataframe of validate
    user_sampler: UnigramTable for users
    item_sampler: UnigramTable for items
    validate_eval: Dataframe for evalutaion on validata data, with columns of col_user, col_item, col_rating
    test_eval: Dataframe for evalutaion on test data, with columns of col_user, col_item, col_rating
    """

    def __init__(
        self,
        train,
        validate,
        test,
        user_fea_type="random",  # can be 'random' 'feature' 'feature_random'
        item_fea_type="random",
        user_fea_dict=None,
        item_fea_dict=None,
        n_neg_test=100,
        col_item=DEFAULT_ITEM_COL,
        col_user=DEFAULT_USER_COL,
        col_rating=DEFAULT_RATING_COL,
        col_flag=DEFAULT_FLAG_COL,
        col_timestamp=DEFAULT_TIMESTAMP_COL,
        seed=12345,
    ):
        """
        Constructor
        """

        # initialize user and item index
        self.user_idx = None
        self.item_idx = None
        self.n_users = 0
        self.n_items = 0
        # set negative sampling for training and test
        self.n_neg_test = n_neg_test
        self.user_fea_type = user_fea_type
        self.item_fea_type = item_fea_type
        self.user_fea_dict = user_fea_dict
        self.item_fea_dict = item_fea_dict
        # get col name of user, item and rating
        self.col_user = col_user
        self.col_item = col_item
        self.col_rating = col_rating
        self.col_timestamp = col_timestamp
        # data preprocessing for training and test data
        self.train = self._data_processing(train, test[0])
        self.validate = self._reindex_list(validate)
        self.test = self._reindex_list(test)
        self.item_sampler = UnigramTable(
            self.train[self.col_item].value_counts().to_dict()
        )
        self.user_sampler = UnigramTable(
            self.train[self.col_user].value_counts().to_dict()
        )
        #         self.user_sampler = UnigramTable(
        #             self.train[self.col_user].value_counts().to_dict()
        #         )
        """
        If the input dataset doesn't contain negative samples, you can call these two methods to feed_neg_sample.
        """
        #         self.validate_eval = self.feed_neg_sample(self.validate)
        #         self.test_eval = self.feed_neg_sample(self.test)
        random.seed(seed)

    def _data_processing(self, train, test, implicit=True):
        """ process the dataset to reindex userID and itemID, also set rating as implicit feedback

        Parameters:
            train (pandas.DataFrame): training data with at least columns (col_user, col_item, col_rating) 
            test (pandas.DataFrame): test data with at least columns (col_user, col_item, col_rating)
                    test can be None, if so, we only process the training data
            implicit (bool): if true, set rating>0 to rating = 1 

        Returns:
            list: train and test pandas.DataFrame Dataset, which have been reindexed.
        
        """

        users_ser = list(
            set(train[self.col_user].unique().flatten()).intersection(
                set(test[self.col_user].unique().flatten())
            )
        )
        items_ser = list(
            set(train[self.col_item].unique().flatten()).intersection(
                set(test[self.col_item].unique().flatten())
            )
        )
        self.n_users = -1
        self.n_items = -1

        while self.n_users != len(users_ser) or self.n_items != len(items_ser):

            test = test[
                test[self.col_user].isin(users_ser)
                & test[self.col_item].isin(items_ser)
            ]
            train = train[
                train[self.col_user].isin(users_ser)
                & train[self.col_item].isin(items_ser)
            ]

            print("n_users reduce from", self.n_users, " to:", len(users_ser))
            print("n_items reduce from", self.n_items, " to:", len(items_ser))

            self.n_users = len(users_ser)
            self.n_items = len(items_ser)

            users_ser = list(
                set(train[self.col_user].unique().flatten()).intersection(
                    set(test[self.col_user].unique().flatten())
                )
            )
            items_ser = list(
                set(train[self.col_item].unique().flatten()).intersection(
                    set(test[self.col_item].unique().flatten())
                )
            )
        self.user_pool = users_ser
        self.item_pool = items_ser

        # Reindex user and item index
        if self.user_idx is None:
            # Map user id
            self.user2id = dict(zip(np.array(self.user_pool), np.arange(self.n_users)))
            self.id2user = {self.user2id[k]: k for k in self.user2id}
        if self.item_idx is None:
            # Map item id
            self.item2id = dict(zip(np.array(self.item_pool), np.arange(self.n_items)))
            self.id2item = {self.item2id[k]: k for k in self.item2id}

        if self.user_fea_type == "random":
            self.user_feature = self.get_random_rep(self.n_users, 512)
        elif self.user_fea_type == "feature":
            self.user_feature = np.array(
                [self.user_fea_dict[self.id2user[k]] for k in np.arange(self.n_users)]
            )
        elif (
            self.user_fea_type == "random_feature"
            or self.user_fea_type == "feature_random"
        ):
            self.user_feature = np.concatenate(
                (
                    np.array(
                        [
                            self.user_fea_dict[self.id2user[k]]
                            for k in np.arange(self.n_users)
                        ]
                    ),
                    self.get_random_rep(self.n_users, 512),
                ),
                axis=1,
            )
            
        if self.item_fea_type == "random":
            self.item_feature = self.get_random_rep(self.n_items, 512)
        elif self.item_fea_type == "feature":
            self.item_feature = np.array(
                [self.item_fea_dict[self.id2item[k]] for k in np.arange(self.n_items)]
            )
        elif (
            self.item_fea_type == "random_feature"
            or self.item_fea_type == "feature_random"
        ):
            self.item_feature = np.concatenate(
                (
                    np.array(
                        [
                            self.item_fea_dict[self.id2item[k]]
                            for k in np.arange(self.n_items)
                        ]
                    ),
                    self.get_random_rep(self.n_items, 512),
                ),
                axis=1,
            )

        return self._reindex(train, implicit)

    def _reindex_list(self, df_list):
        """
        _reindex for list of dataset. For example, validate and test can be a list for evaluation
        
        """
        df_list_new = []
        for df in df_list:
            df = df[
                df[self.col_user].isin(self.user_pool)
                & df[self.col_item].isin(self.item_pool)
            ]
            df_list_new.append(self._reindex(df))
        return df_list_new

    def _reindex(self, df, implicit=True):
        """ 
        Process dataset to reindex userID and itemID, also set rating as implicit feedback

        Parameters:
            df (pandas.DataFrame): dataframe with at least columns (col_user, col_item, col_rating) 
            implicit (bool): if true, set rating>0 to rating = 1 

        Returns:
            list: train and test pandas.DataFrame Dataset, which have been reindexed.
        
        """

        # If testing dataset is None
        if df is None:
            return None

        # Map user_idx and item_idx
        df[self.col_user] = df[self.col_user].apply(lambda x: self.user2id[x])
        df[self.col_item] = df[self.col_item].apply(lambda x: self.item2id[x])

        #         df = pd.merge(df, self.user_idx, on=self.col_user, how="left")
        #         df = pd.merge(df, self.item_idx, on=self.col_item, how="left")

        # If implicit feedback, set rating as 1.0 or 0.0
        if implicit:
            df[self.col_rating] = df[self.col_rating].apply(lambda x: float(x > 0))

        # Select relevant columns
        #         df_reindex = df[
        #             [self.col_user + "_idx", self.col_item + "_idx", self.col_rating]
        #         ]
        #         df_reindex.columns = [self.col_user, self.col_item, self.col_rating]

        return df

    def get_random_rep(self, raw_num, dim):
        return np.random.normal(size=(raw_num, dim))

    # get binary identity representation of items or users, instead of np.eye with too large memory
    def get_bin_rep(self, number):
        width = int(math.ceil(np.log2(number)))
        bin_rep = []
        for _i in range(number):
            bin_rep.append([int(x) for x in np.binary_repr(_i, width=width)])
        return np.asarray(bin_rep)

    def feed_neg_sample(self, eval_df):
        """ 
        sampling negative sampling for evaluation.


        Parameters:
                eval_df: Dataframe with column naming 'user_ids', 'item_ids' and 'ratings',
                where all the ratings is 1

        Returns:
                eval_df with column naming 'user_ids', 'item_ids' and 'ratings' appended 
                with negetive samples and the ratings is 0
        """
        print("sampling negative items...")
        interact_status = (
            eval_df.groupby(["user_ids"])["item_ids"].apply(set).reset_index()
        )
        total_interact = pd.DataFrame(
            {"user_ids": [], "item_ids": [], "rating": []}, dtype=np.int32
        )
        negative_num = self.n_neg_test
        for index, user_items in interact_status.iterrows():
            u = user_items["user_ids"]
            items = set(user_items["item_ids"])  # item set for user u
            n_items = len(items)  # number of positive item for user u
            sample_neg_items = set(
                self.item_sampler.sample(negative_num + n_items, 1, True)
            )  # first sample negative_num+n_items items
            sample_neg_items = list(sample_neg_items - items)[:negative_num]
            # filter the positive items and truncate the first negative_num
            #     print(len(sample_neg_items))
            tp_items = np.append(list(items), sample_neg_items)
            #     print(len(tp_items))

            tp_users = np.ones(negative_num + n_items, dtype=np.int32) * u
            tp_ones = np.ones(n_items, dtype=np.int32)
            tp_zeros = np.zeros(negative_num, dtype=np.int32)
            ratings = np.append(tp_ones, tp_zeros)
            #     print(len(tp_users)),print(len(tp_items)),print(len(ratings))
            tp = pd.DataFrame(
                {"user_ids": tp_users, "item_ids": tp_items, "ratings": ratings}
            )
            total_interact = total_interact.append(tp)

        total_interact = sklearn.utils.shuffle(total_interact)
        return total_interact

    def train_loader(self, batch_size, shuffle=True):
        """ 
        feed train data every batch
        Parameters:
                batch size (int)
                shuffle (bool): if true, train data will be shuffled
        Returns:
                list: userID list, itemID list, rating list.
                public data loader return the userID, itemID consistent with raw data

        """

        # yield batch of training data with `shuffle`

        indices = np.arange(len(self.users))
        if shuffle:
            random.shuffle(indices)
        for i in range(len(indices) // batch_size):
            begin_idx = i * batch_size
            end_idx = (i + 1) * batch_size
            batch_indices = indices[begin_idx:end_idx]

            # train_loader() could be called and used by our users in other situations,
            # who expect the not re-indexed data. So we convert id --> original user and item
            # when returning batch

            yield [
                [self.id2user[x] for x in self.users[batch_indices]],
                [self.id2item[x] for x in self.items[batch_indices]],
                self.ratings[batch_indices],
            ]

    def evaluate_all(self, data_list, model, k=0, t=0):
        """ 
        evaluate the performance for all the validate set or test set.


        Parameters:
                data: Dataframe with column naming 'user_ids', 'item_ids' and 'ratings'
                model: the trained model, which must have a 'predict(user_ids, item_ids)'
                    method that can return the corresponding rating scores for user-item pairs
                k: if k=0, will return the result for k=5,10,20.

        Returns:
                return the evaluation scores of the following metrics scores:MAP,NDCG,
                Precision,Recall on value of k.
                
                example:
                {'MAP@5': 0.5,
                 'NDCG@5': 0.5,
                 'Precision@5': 0.5,
                 'Recall@5':0.5
                 }
        """
        result_list = []
        for data in data_list:
            result = self.evaluate(data, model, k, t)
            result_list.append(result)
        return result_list

    def evaluate(self, data, model, k=0, t=0):
        """ 
        evaluate the performance for all the validate set or test set.


        Parameters:
                data: Dataframe with column naming 'user_ids', 'item_ids' and 'ratings'
                model: the trained model, which must have a 'predict(user_ids, item_ids)'
                    method that can return the corresponding rating scores for user-item pairs
                k: if k=0, will return the result for k=5,10,20.

        Returns:
                return the evaluation scores of the following metrics scores:MAP,NDCG,
                Precision,Recall on value of k.
                
                example:
                {'MAP@5': 0.5,
                 'NDCG@5': 0.5,
                 'Precision@5': 0.5,
                 'Recall@5':0.5
                 }
        """
        user_ids = data["user_ids"].to_numpy()
        item_ids = data["item_ids"].to_numpy()
        ratings = data["ratings"].to_numpy()
        #         print(len(data.index))
        if t == 0:
            prediction = np.array(model.predict(user_ids, item_ids))
        else:
            prediction = np.array(model.predict(user_ids, item_ids, t))
        pred = pd.DataFrame(
            {"col_user": user_ids, "col_item": item_ids, "prediction": prediction}
        )
        test_df = pd.DataFrame(
            {"col_user": user_ids, "col_item": item_ids, "col_rating": ratings}
        )

        result = {}

        if k == 0:
            TOP_K = 5
            result["map@5"] = map_at_k(
                test_df, pred, col_prediction="prediction", k=TOP_K
            )
            result["ndcg@5"] = ndcg_at_k(
                test_df, pred, col_prediction="prediction", k=TOP_K
            )
            result["precision@5"] = precision_at_k(
                test_df, pred, col_prediction="prediction", k=TOP_K
            )
            result["recall@5"] = recall_at_k(
                test_df, pred, col_prediction="prediction", k=TOP_K
            )

            TOP_K = 10
            result["map@10"] = map_at_k(
                test_df, pred, col_prediction="prediction", k=TOP_K
            )
            result["ndcg@10"] = ndcg_at_k(
                test_df, pred, col_prediction="prediction", k=TOP_K
            )
            result["precision@10"] = precision_at_k(
                test_df, pred, col_prediction="prediction", k=TOP_K
            )
            result["recall@10"] = recall_at_k(
                test_df, pred, col_prediction="prediction", k=TOP_K
            )

            TOP_K = 20
            result["map@20"] = map_at_k(
                test_df, pred, col_prediction="prediction", k=TOP_K
            )
            result["ndcg@20"] = ndcg_at_k(
                test_df, pred, col_prediction="prediction", k=TOP_K
            )
            result["precision@20"] = precision_at_k(
                test_df, pred, col_prediction="prediction", k=TOP_K
            )
            result["recall@20"] = recall_at_k(
                test_df, pred, col_prediction="prediction", k=TOP_K
            )
        else:
            result["map@" + str(k)] = map_at_k(
                test_df, pred, col_prediction="prediction", k=k
            )
            result["ndcg@" + str(k)] = ndcg_at_k(
                test_df, pred, col_prediction="prediction", k=k
            )
            result["precision@" + str(k)] = precision_at_k(
                test_df, pred, col_prediction="prediction", k=k
            )
            result["recall@" + str(k)] = recall_at_k(
                test_df, pred, col_prediction="prediction", k=k
            )
        #         print(result)
        return result

    def evaluate_vali(self, data, model, k=0, t=0):
        """ 
        evaluate the performance for all the validate set or test set.


        Parameters:
                data: Dataframe with column naming 'user_ids', 'item_ids' and 'ratings'
                model: the trained model, which must have a 'predict(user_ids, item_ids)'
                    method that can return the corresponding rating scores for user-item pairs
                k: if k=0, will return the result for k=5,10,20.

        Returns:
                return the evaluation scores of the following metrics scores:MAP,NDCG,
                Precision,Recall on value of k.
                
                example:
                {'MAP@5': 0.5,
                 'NDCG@5': 0.5,
                 'Precision@5': 0.5,
                 'Recall@5':0.5
                 }
        """
        user_ids = data["user_ids"].to_numpy()
        item_ids = data["item_ids"].to_numpy()
        ratings = data["ratings"].to_numpy()
        #         print(len(data.index))
        if t == 0:
            prediction = np.array(model.predict(user_ids, item_ids))
        else:
            prediction = np.array(model.predict(user_ids, item_ids, t))
        pred = pd.DataFrame(
            {"col_user": user_ids, "col_item": item_ids, "prediction": prediction}
        )
        test_df = pd.DataFrame(
            {"col_user": user_ids, "col_item": item_ids, "col_rating": ratings}
        )

        result = {}

        TOP_K = 10
        result["ndcg@10"] = ndcg_at_k(
            test_df, pred, col_prediction="prediction", k=TOP_K
        )
        result["recall@10"] = recall_at_k(
            test_df, pred, col_prediction="prediction", k=TOP_K
        )
        #         print(result)
        return result

    def evaluate_all_mean(self, data_list, model, k=0):
        result_list = self.evaluate_all(data_list, model)
        n = len(result_list)
        sum_result = {}
        sum_result["map@5"] = 0
        sum_result["ndcg@5"] = 0
        sum_result["precision@5"] = 0
        sum_result["recall@5"] = 0
        sum_result["map@10"] = 0
        sum_result["ndcg@10"] = 0
        sum_result["precision@10"] = 0
        sum_result["recall@10"] = 0
        sum_result["map@20"] = 0
        sum_result["ndcg@20"] = 0
        sum_result["precision@20"] = 0
        sum_result["recall@20"] = 0
        for result in result_list:
            sum_result["map@5"] += result["map@5"]
            sum_result["ndcg@5"] += result["ndcg@5"]
            sum_result["precision@5"] += result["precision@5"]
            sum_result["recall@5"] += result["recall@5"]
            sum_result["map@10"] += result["map@10"]
            sum_result["ndcg@10"] += result["ndcg@10"]
            sum_result["precision@10"] += result["precision@10"]
            sum_result["recall@10"] += result["recall@10"]
            sum_result["map@20"] += result["map@20"]
            sum_result["ndcg@20"] += result["ndcg@20"]
            sum_result["precision@20"] += result["precision@20"]
            sum_result["recall@20"] += result["recall@20"]

        sum_result["map@5"] /= n
        sum_result["ndcg@5"] /= n
        sum_result["precision@5"] /= n
        sum_result["recall@5"] /= n
        sum_result["map@10"] /= n
        sum_result["ndcg@10"] /= n
        sum_result["precision@10"] /= n
        sum_result["recall@10"] /= n
        sum_result["map@20"] /= n
        sum_result["ndcg@20"] /= n
        sum_result["precision@20"] /= n
        sum_result["recall@20"] /= n
        return sum_result


def load_instacart(data_base_dir=instacart_dir, percent=0.1, abs_dir=None):
    """
    Load Instacart dataset based on the percentage of users and items.
    
    Parameters
    ----------
    data_base_dir: Base dir of the dataset.
    
    Returns
    -------
    Dataframe of the Instacart data, with columns: USER_COL, ORDER_COL, ITEM_COL
    RATING_COL, TIMESTAMP_COL, and FLAG_COL
    
    
    """
    suff_str = "_0.1"
    if percent == 1:
        suff_str = "1_1"
    elif percent == 0.05:
        suff_str = "0.05_0.05"
    elif percent == 0.1:
        suff_str = "0.1_0.1"
    elif percent == 0.25:
        suff_str = "0.25_0.25"
    elif percent == 0.5:
        suff_str = "0.5_0.5"
    elif percent == 0.75:
        suff_str = "0.75_0.75"
    else:
        print("ERROR: unsupported percent")

    print("Loading data from:", data_base_dir + suff_str)
    if abs_dir == None:
        date_dir = data_base_dir + str(suff_str)
    else:
        date_dir = abs_dir
    loaded = np.load(date_dir + "_train.npz")
    train_df = pd.DataFrame(
        data={
            DEFAULT_USER_COL: loaded[DEFAULT_USER_COL],
            DEFAULT_ORDER_COL: loaded[DEFAULT_ORDER_COL],
            DEFAULT_ITEM_COL: loaded[DEFAULT_ITEM_COL],
            DEFAULT_RATING_COL: loaded[DEFAULT_RATING_COL],
            DEFAULT_TIMESTAMP_COL: loaded[DEFAULT_TIMESTAMP_COL],
            DEFAULT_FLAG_COL: loaded[DEFAULT_FLAG_COL],
        }
    )
    valid_dfs = []
    test_dfs = []
    for i in range(10):
        loaded = np.load(date_dir + "_valid_" + str(i) + ".npz")
        valid_df = pd.DataFrame(
            data={
                DEFAULT_USER_COL: loaded[DEFAULT_USER_COL],
                DEFAULT_ITEM_COL: loaded[DEFAULT_ITEM_COL],
                DEFAULT_RATING_COL: loaded[DEFAULT_RATING_COL],
            }
        )
        loaded = np.load(date_dir + "_test_" + str(i) + ".npz")
        test_df = pd.DataFrame(
            data={
                DEFAULT_USER_COL: loaded[DEFAULT_USER_COL],
                DEFAULT_ITEM_COL: loaded[DEFAULT_ITEM_COL],
                DEFAULT_RATING_COL: loaded[DEFAULT_RATING_COL],
            }
        )
        valid_dfs.append(valid_df)
        test_dfs.append(test_df)
    return train_df, valid_dfs, test_dfs


def load_other_data(data_dir):
    loaded = np.load(data_dir + "train.npz")
    train_df = pd.DataFrame(
        data={
            DEFAULT_USER_COL: loaded[DEFAULT_USER_COL],
            DEFAULT_ORDER_COL: loaded[DEFAULT_ORDER_COL],
            DEFAULT_ITEM_COL: loaded[DEFAULT_ITEM_COL],
            DEFAULT_RATING_COL: loaded[DEFAULT_RATING_COL],
            DEFAULT_TIMESTAMP_COL: loaded[DEFAULT_TIMESTAMP_COL],
        }
    )
    valid_dfs = []
    test_dfs = []
    for i in range(10):
        loaded = np.load(data_dir + "valid_" + str(i) + ".npz")
        valid_df = pd.DataFrame(
            data={
                DEFAULT_USER_COL: loaded[DEFAULT_USER_COL],
                DEFAULT_ITEM_COL: loaded[DEFAULT_ITEM_COL],
                DEFAULT_RATING_COL: loaded[DEFAULT_RATING_COL],
            }
        )
        loaded = np.load(data_dir + "test_" + str(i) + ".npz")
        test_df = pd.DataFrame(
            data={
                DEFAULT_USER_COL: loaded[DEFAULT_USER_COL],
                DEFAULT_ITEM_COL: loaded[DEFAULT_ITEM_COL],
                DEFAULT_RATING_COL: loaded[DEFAULT_RATING_COL],
            }
        )
        valid_dfs.append(valid_df)
        test_dfs.append(test_df)
    return train_df, valid_dfs, test_dfs


def load_dataset(data_str="instacart", percent=1):
    """
    Load dataset based on the percentage of users and items.
    
    Parameters
    ----------
    data_str: instacart,tafeng,dunnhumby
    
    Returns
    -------
    Dataframe of the Instacart data, with columns: USER_COL, ORDER_COL, ITEM_COL
    RATING_COL, TIMESTAMP_COL, and FLAG_COL
    
    
    """
    print("Loading dataset :", data_str)
    if percent != 1:
        print("Percentage of dataset :", percent)
    if data_str == "instacart":
        return load_instacart(percent=percent)
    elif data_str == "tafeng":
        return load_other_data(tafeng_dir)
    elif data_str == "dunnhumby":
        return load_other_data(dunnhumby_dir)


def load_item_fea(base_dir, fea_type="one_hot"):
    """
    Load item_feature.csv
    
    Returns
    -------
    item_feature dict.
        
    """
    print("load_item_fea...")
    if fea_type == "one_hot":
        item_feature_file = open(base_dir + "raw/item_feature.csv", "r")
        item_feature = {}
        lines = item_feature_file.readlines()
        for index in range(1, len(lines)):
            key_value = lines[index].split(",")
            item_id = int(key_value[0])
            feature = np.array(key_value[1].split(" "), dtype=np.float)
            item_feature[item_id] = feature
        return item_feature
    elif fea_type == "word2vec":
        item_feature_file = open(base_dir + "raw/item_feature_w2v.csv", "r")
        item_feature = {}
        lines = item_feature_file.readlines()
        for index in range(1, len(lines)):
            key_value = lines[index].split(",")
            item_id = int(key_value[0])
            feature = np.array(key_value[1].split(" "), dtype=np.float)
            item_feature[item_id] = feature
        return item_feature
    elif fea_type == "bert":
        item_feature_file = open(base_dir + "raw/item_feature_bert.csv", "r")
        item_feature = {}
        lines = item_feature_file.readlines()
        for index in range(1, len(lines)):
            key_value = lines[index].split(",")
            item_id = int(key_value[0])
            feature = np.array(key_value[1].split(" "), dtype=np.float)
            item_feature[item_id] = feature
        return item_feature
    elif fea_type == "word2vec_one_hot" or fea_type == "one_hot_word2vec":
        item_fea_dict1 = load_item_fea(base_dir=base_dir, fea_type="word2vec")
        item_fea_dict2 = load_item_fea(base_dir=base_dir, fea_type="one_hot")
        item_fea_dict = {}
        for key, _ in item_fea_dict1.items():
            item_fea_dict[key] = np.concatenate(
                (item_fea_dict1[key], item_fea_dict2[key])
            )
        return item_fea_dict
    elif (
        fea_type == "word2vec_one_hot_bert"
        or fea_type == "bert_one_hot_word2vec"
        or fea_type == "one_hot_bert_word2vec"
    ):
        item_fea_dict1 = load_item_fea(base_dir=base_dir, fea_type="word2vec")
        item_fea_dict2 = load_item_fea(base_dir=base_dir, fea_type="one_hot")
        item_fea_dict3 = load_item_fea(base_dir=base_dir, fea_type="bert")
        item_fea_dict = {}
        for key, _ in item_fea_dict1.items():
            item_fea_dict[key] = np.concatenate(
                (item_fea_dict1[key], item_fea_dict2[key])
            )
            item_fea_dict[key] = np.concatenate(
                (item_fea_dict[key], item_fea_dict3[key])
            )
        return item_fea_dict

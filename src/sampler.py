"""
Created on Aug 5, 2019
Updated on XX,2019 BY xxx@

Classes describing datasets of user-item interactions. Instances of these
are returned by dataset fetching and dataset pre-processing functions.

@author: Zaiqiao Meng (zaiqiao.meng@gmail.com)

"""

import numpy as np
import sys
from tqdm import tqdm
import os


class Sampler(object):
    def __init__(self, dataTrain, percent, N_SAMPLE, sample_dir, dump=True):
        self.sample_file = (
            sample_dir + "triple_" + str(percent) + "_" + str(N_SAMPLE) + ".csv"
        )
        self.dataTrain = dataTrain
        self.sample_dir = sample_dir
        self.percent = percent
        self.N_SAMPLE = N_SAMPLE
        self.dump = dump
        if not os.path.exists(sample_dir):
            os.mkdir(sample_dir)
        print("successfully initialized!")
        sys.stdout.flush()

    def sample(self):
        print("preparing training triples ... ")
        sys.stdout.flush()
        print("current progress for", self.N_SAMPLE, "samples: ", end=" ")
        sys.stdout.flush()
        n_orders = self.dataTrain.shape[0]
        sampled_index = np.random.choice(n_orders, size=self.N_SAMPLE)
        sampled_order = self.dataTrain.iloc[sampled_index].reset_index()

        process_bar = tqdm(range(self.N_SAMPLE))
        res = []
        for i in process_bar:
            _index, _uid, _tid, _items = sampled_order.iloc[i]
            #             if len(_items)<2:
            #                 _ ,_, _items_a = self.dataTrain.iloc[_index-1]
            #                 _ ,_, _items_b = self.dataTrain.iloc[_index+1]
            #                 _items.extend(_items_a)
            #                 _items.extend(_items_b)
            _i, _j = np.random.choice(_items, size=2)
            res.append([_uid, _i, _j])
        print("done!")
        res = np.array(res)
        if self.dump:
            np.savetxt(self.sample_file, res, delimiter=", ")
        return res

def load_triples_from_file(sample_file):
    print("load_triples_from_file:", sample_file)
    res = np.genfromtxt(sample_file, delimiter=", ")
    return res
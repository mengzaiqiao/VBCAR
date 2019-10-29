"""
Created on Aug 5, 2019
Updated on XX,2019 BY xxx@

Classes describing datasets of user-item interactions. Instances of these
are returned by dataset fetching and dataset pre-processing functions.

@author: Zaiqiao Meng (zaiqiao.meng@gmail.com)

"""

import numpy as np
import pandas as pd
import pickle
import argparse

import torch.optim as optim
from torch import Tensor
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from livelossplot import PlotLosses
import GPUtil
import os
import sys
import time

sys.path.append("../")

import random

base_string = "abcdefghijklmnopqrstuvwxyz"

from src.vbcar import VBCAR
from src import sampler
from src.sampler import Sampler

from src import data_util
from src.monitor import Monitor
# from src import logger


def parse_args():
    parser = argparse.ArgumentParser(description="Run VBCR.")
    parser.add_argument(
        "--DATASET", nargs="?", type=str, default="dunnhumby", help="DATASET"
    )
    parser.add_argument("--PERCENT", nargs="?", type=float, default=1, help="PERCENT")
    parser.add_argument(
        "--N_SAMPLE", nargs="?", type=int, default=1000000, help="N_SAMPLE"
    )
    parser.add_argument(
        "--I_FEA_TYPE",
        nargs="?",
        type=str,
        default="one_hot",
        help="item feature type, can be 'one_hot' 'word2vec' 'one_hot+word2vec'",
    )
    parser.add_argument(
        "--FEA_CONSTR",
        nargs="?",
        type=str,
        default="random",
        help="to construct feature according to the identical feature and the item feature, can be 'random' 'feature' or 'random+feature'",
    )
    parser.add_argument("--MODEL", nargs="?", type=str, default="VAE", help="MODEL")
    parser.add_argument("--EMB_DIM", nargs="?", type=int, default=64, help="EMB_DIM")
    parser.add_argument("--LAT_DIM", nargs="?", type=int, default=512, help="LAT_DIM")
    parser.add_argument(
        "--INIT_LR", nargs="?", type=float, default=0.002, help="INIT_LR"
    )
    parser.add_argument(
        "--BATCH_SIZE", nargs="?", type=int, default=400, help="BATCH_SIZE"
    )
    parser.add_argument("--OPTI", nargs="?", type=str, default="RMSprop", help="OPTI")
    parser.add_argument("--ALPHA", nargs="?", type=float, default=0.01, help="ALPHA")
    parser.add_argument("--EPOCH", nargs="?", type=int, default=100, help="EPOCH")
    parser.add_argument(
        "--RESULT_FILE",
        nargs="?",
        type=str,
        default="result_VBCR_test.csv",
        help="RESULT_FILE",
    )
    parser.add_argument("--REMARKS", nargs="?", type=str, default="", help="REMARKS")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    PERCENT = args.PERCENT
    DATASET = args.DATASET
    I_FEA_TYPE = args.I_FEA_TYPE
    FEA_CONSTR = args.FEA_CONSTR
    N_SAMPLE = args.N_SAMPLE
    MODEL = args.MODEL
    EMB_DIM = args.EMB_DIM
    LAT_DIM = args.LAT_DIM
    INIT_LR = args.INIT_LR
    BATCH_SIZE = args.BATCH_SIZE
    OPTI = args.OPTI
    ALPHA = args.ALPHA
    EPOCH = args.EPOCH
    RESULT_FILE = args.RESULT_FILE
    REMARKS = args.REMARKS

    output_dir = "./result/" + DATASET + "/"
    sample_dir = "./sample/" + DATASET + "/"
    embedding_dir = "./embedding/" + DATASET + "/"
    log_dir = "./log/" + DATASET + "/"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    if not os.path.exists(sample_dir):
        os.mkdir(sample_dir)
    if not os.path.exists(embedding_dir):
        os.mkdir(embedding_dir)
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    timestamp_str = str(int(time.time()))

    para_str = (
        DATASET
        + "_"
        + str(PERCENT)
        + "_"
        + str(N_SAMPLE)
        + "_"
        + str(MODEL)
        + "_"
        + str(I_FEA_TYPE)
        + "_"
        + str(FEA_CONSTR)
        + "_"
        + str(EMB_DIM)
        + "_"
        + str(LAT_DIM)
        + "_"
        + str(INIT_LR)
        + "_"
        + OPTI
        + "_"
        + str(ALPHA)
    )
    model_str = para_str + "_" + timestamp_str

    """
    Logging
    """
#     logging = logger.init_std_logger(log_dir + model_str)

    """
    file paths to be saved
    """
    REMARKS = model_str + " : " + timestamp_str
    print("timestamp_str:", timestamp_str)
    print("REMARKS:", REMARKS)

    result_file = output_dir + RESULT_FILE
    print("result will save in file:", result_file)
    model_save_dir = embedding_dir + model_str + ".pt"
    print("model will save in file:", model_save_dir)
    embedding_save_dir = embedding_dir + model_str
    print("embedding will save in file: ", embedding_save_dir)

    print(args)
    print(sys.version)
    print("pytorch version:", torch.__version__)

    """
    Loading dataset
    """
    train, test, validate = data_util.load_dataset(data_str=DATASET, percent=PERCENT)
#     feat_base_dir = ''
#     item_fea_dict = data_util.load_item_fea(base_dir=feat_base_dir, fea_type=I_FEA_TYPE)

    print(len(train.index), len(validate[0].index), len(test[0].index))

    data = data_util.Dataset(
        train=train,
        validate=validate,
        test=test,
#         item_fea_dict=item_fea_dict,
        user_fea_type="random",
        item_fea_type=FEA_CONSTR,
    )
    n_users = data.n_users
    n_items = data.n_items

    dataTrain = (
        data.train.groupby(["user_ids", "order_ids"])["item_ids"]
        .apply(list)
        .reset_index()
    )
    dataTrain.rename(
        columns={"user_ids": "UID", "order_ids": "TID", "item_ids": "PID"}, inplace=True
    )

    """
    Sample triples or load triples samples from files
    """
    sample_file = sample_dir + "triple_" + str(PERCENT) + "_" + str(N_SAMPLE) + ".csv"
    my_sampler = Sampler(dataTrain, PERCENT, N_SAMPLE, sample_dir)
    train_Triples = my_sampler.sample()

    """
    Get a gpu id list sorted by the most available memory
    """
    DEVICE_ID_LIST = GPUtil.getAvailable(
        order="memory", limit=3
    )  # get the fist gpu with the lowest load

    """
    Find a gpu device
    """
    if len(DEVICE_ID_LIST) < 1:
        gpu_id = None
        device_str = "cpu"
    else:
        gpu_id = DEVICE_ID_LIST[0]
        #         os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
        device_str = "cuda:" + str(DEVICE_ID_LIST[0])

    """
    monitoring resources of this application
    """
    monitor = Monitor(1, gpu_id, os.getpid())

    data_loader = DataLoader(
        Tensor(train_Triples), batch_size=BATCH_SIZE, shuffle=True, drop_last=True
    )

    """
    init model
    """
    print("init model ", MODEL)
    vbcr = VBCAR(
        data_loader,
        data,
        model_save_dir,
        n_users,
        n_items,
        n_neg=5,
        emb_dim=EMB_DIM,
        latent_dim=LAT_DIM,
        batch_size=BATCH_SIZE,
        initial_lr=INIT_LR,
        activator="tanh",
        iteration=EPOCH,
        optimizer_type=OPTI,
        alpha=ALPHA,
        model_str=MODEL,
        show_result=True,
        device_str=device_str,
    )
    print("strat traning... ")
    vbcr.train()
    #     vbcr.save_embedding(embedding_save_dir)
    run_time = monitor.stop()

    """
    Prediction and evalution on test set
    """
    columns = [
        "PERCENT",
        "N_SAMPLE",
        "MODEL",
        "EMB_DIM",
        "INIT_LR",
        "BATCH_SIZE",
        "OPTI",
        "ALPHA",
        "time",
    ]
    result_para = {
        "PERCENT": [PERCENT],
        "N_SAMPLE": [N_SAMPLE],
        "MODEL": [MODEL],
        "LAT_DIM": [LAT_DIM],
        "EMB_DIM": [EMB_DIM],
        "INIT_LR": [INIT_LR],
        "BATCH_SIZE": [BATCH_SIZE],
        "OPTI": [OPTI],
        "ALPHA": [ALPHA],
        "REMARKS": [REMARKS],
    }

    for i in range(10):
        result = data.evaluate(data.test[i], vbcr.load_best_model())
        print(result)
        result["time"] = [run_time]
        result.update(result_para)
        result_df = pd.DataFrame(result)

        if os.path.exists(result_file):
            print(result_file, " already exists, appending result to it")
            total_result = pd.read_csv(result_file)
            for column in columns:
                if column not in total_result.columns:
                    total_result[column] = "-"
            total_result = total_result.append(result_df)
        else:
            print("create new result_file:", result_file)
            total_result = result_df

        total_result.to_csv(result_file, index=False)
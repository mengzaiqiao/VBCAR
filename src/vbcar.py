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
import copy
from sklearn.utils import shuffle

import torch.optim as optim
from torch import Tensor
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim

from src.models import Triple2vec, Triple2vec_Single
from src.models import VAE
from tqdm import tqdm
from livelossplot import PlotLosses
from src.sampler import Sampler


class VBCAR:
    def __init__(
        self,
        data_loader,
        data,
        output_file_name,
        n_users,
        n_items,
        n_neg=5,
        emb_dim=32,
        latent_dim=32,
        batch_size=128,
        iteration=100,
        optimizer_type="Adam",
        initial_lr=0.00025,
        activator="tanh",
        alpha=0.0,
        model_str="VAE2item",
        show_result=False,
        device_str="cpu",
    ):
        self.device = torch.device(device_str)
        self.data_loader = data_loader
        self.data = data
        self.model_str = model_str
        self.output_file_name = output_file_name
        self.n_users = n_users
        self.n_items = n_items
        self.emb_dim = emb_dim
        self.n_neg = n_neg
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.iteration = iteration
        self.initial_lr = initial_lr
        self.activator = activator
        self.alpha = alpha
        self.show_result = show_result

        if self.model_str == "VAE2item":
            self.init_fea(self.data)
            self.model = VAE2item(
                self.n_users,
                self.n_items,
                self.user_fea,
                self.item_fea,
                self.latent_dim,
                self.emb_dim,
                self.n_neg,
                self.batch_size,
                self.alpha,
            ).to(self.device)
        elif self.model_str == "VAE":
            self.init_fea(self.data)
            self.model = VAE(
                self.n_users,
                self.n_items,
                self.user_fea,
                self.item_fea,
                self.latent_dim,
                self.emb_dim,
                self.n_neg,
                self.batch_size,
                self.activator,
                self.alpha,
                self.device,
            ).to(self.device)
        elif self.model_str == "VAE_N":
            self.init_fea_noside()
            self.model = VAE(
                self.n_users,
                self.n_items,
                self.user_fea,
                self.item_fea,
                self.latent_dim,
                self.emb_dim,
                self.n_neg,
                self.batch_size,
                self.activator,
                self.alpha,
                self.device,
            ).to(self.device)
        elif self.model_str == "Triple2vec":
            self.model = Triple2vec(
                self.n_users, self.n_items, self.emb_dim, self.n_neg, self.batch_size
            ).to(self.device)
        elif self.model_str == "Triple2vec_Single":
            self.model = Triple2vec_Single(
                self.n_users, self.n_items, self.emb_dim, self.n_neg, self.batch_size
            ).to(self.device)

        if optimizer_type == "RMSprop":
            self.optimizer = optim.RMSprop(self.model.parameters(), lr=self.initial_lr)
        elif optimizer_type == "Adam":
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.initial_lr)
        elif optimizer_type == "SGD":
            self.optimizer = optim.SGD(
                self.model.parameters(), lr=self.initial_lr, momentum=0.9
            )
        print(self.model)

    def init_fea(self, data):
        """
        reserve for construct from data
        """
        self.user_fea = torch.tensor(
            self.data.user_feature,
            dtype=torch.float32,
            requires_grad=False,
            device=self.device,
        )

        self.item_fea = torch.tensor(
            self.data.item_feature,
            dtype=torch.float32,
            requires_grad=False,
            device=self.device,
        )
        print(self.user_fea.shape)

    def get_bin_rep(self, number):
        import math

        width = int(math.ceil(np.log2(number)))
        bin_rep = []
        for _i in range(number):
            bin_rep.append([int(x) for x in np.binary_repr(_i, width=width)])
        return np.asarray(bin_rep)

    def init_fea_noside(self):
        """
        Currently only use the one-hot encoding.
        Will load them from the data files.
        """
        init_fea_dim = 512
        #         self.user_fea = torch.tensor(
        #             np.eye(self.n_users),
        #             dtype=torch.float32,
        #             requires_grad=False,
        #             device=self.device,
        #         )
        #         self.item_fea = torch.tensor(
        #             np.eye(self.n_items),
        #             dtype=torch.float32,
        #             requires_grad=False,
        #             device=self.device,
        #         )
        self.user_fea = torch.randn(
            self.n_users, init_fea_dim, dtype=torch.float32, device=self.device
        )
        self.item_fea = torch.randn(
            self.n_items, init_fea_dim, dtype=torch.float32, device=self.device
        )

        print(self.user_fea.size())
        print(self.item_fea.size())

    def train(self):
        """Multiple training.

        Returns:
            None.
        """
        max_noprogress = 5
        _loss_train_min = 1e-5
        n_noprogress = 0

        process_bar = tqdm(range(self.iteration))
        liveloss = PlotLosses(fig_path=self.output_file_name + ".iter.pdf")
        loss_list = []
        _best_ndcg = 0
        for i in process_bar:
            logs = {}
            all_loss = 0
            kl_loss = 0
            batch_num = 0
            for batch_ndx, sample in enumerate(self.data_loader):
                pos_u = torch.tensor(
                    [triple[0] for triple in sample],
                    dtype=torch.int64,
                    device=self.device,
                )
                pos_i_1 = torch.tensor(
                    [triple[1] for triple in sample],
                    dtype=torch.int64,
                    device=self.device,
                )
                pos_i_2 = torch.tensor(
                    [triple[2] for triple in sample],
                    dtype=torch.int64,
                    device=self.device,
                )

                neg_u = torch.tensor(
                    self.data.user_sampler.sample(self.n_neg, len(sample)),
                    dtype=torch.int64,
                    device=self.device,
                )
                neg_i_1 = torch.tensor(
                    self.data.item_sampler.sample(self.n_neg, len(sample)),
                    dtype=torch.int64,
                    device=self.device,
                )
                neg_i_2 = torch.tensor(
                    self.data.item_sampler.sample(self.n_neg, len(sample)),
                    dtype=torch.int64,
                    device=self.device,
                )
                #                 print(pos_u,neg_u)

                self.optimizer.zero_grad()
                loss = self.model.forward(
                    pos_u, pos_i_1, pos_i_2, neg_u, neg_i_2, neg_i_2
                )
                #                 print(loss)
                loss.backward()
                self.optimizer.step()
                all_loss = all_loss + loss
                kl_loss = kl_loss + self.model.kl_loss
                batch_num = batch_ndx
            if self.device.type == "cuda":
                all_loss = all_loss.cpu()
                if kl_loss != 0:
                    kl_loss = kl_loss.cpu()

            logs["loss"] = all_loss.item() / batch_num

            if self.show_result:
                data_i = np.random.randint(10)
                result = self.data.evaluate_vali(self.data.test[data_i], self.model)
                logs["ndcg@10_test"], logs["recall@10_test"] = (
                    result["ndcg@10"],
                    result["recall@10"],
                )
                result = self.data.evaluate_vali(self.data.validate[data_i], self.model)
                logs["ndcg@10_val"], logs["recall@10_val"] = (
                    result["ndcg@10"],
                    result["recall@10"],
                )
                if _best_ndcg < result["ndcg@10"]:
                    _best_ndcg = result["ndcg@10"]
                    self.best_model = copy.deepcopy(self.model.state_dict())
                    torch.save(self.best_model, self.output_file_name)
            if kl_loss != 0:
                logs["kl_loss"] = kl_loss.item() / batch_num
                logs["loss"] = logs["loss"] - logs["kl_loss"]

            loss_list.append(logs["loss"])

            if i > 1:
                if abs(loss_list[i] - loss_list[i - 1]) < _loss_train_min:
                    n_noprogress += 1
                else:
                    n_noprogress = 0

            liveloss.update(logs)
            liveloss.draw()
            process_bar.set_description(
                "Loss: %0.8f, lr: %0.6f"
                % (logs["loss"], self.optimizer.param_groups[0]["lr"])
            )
            print("=== #no progress: ", n_noprogress)

            if n_noprogress >= max_noprogress:
                liveloss.draw()
                break

            """Sets the learning rate to the initial LR decayed by 10 every 10 epochs"""
            lr = self.initial_lr * (0.5 ** (i // 10))
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr
            if i >= self.iteration - 1:
                liveloss.draw()

    def save_embedding(self, file_path):
        self.model.save_embedding(self.data.id2user, self.data.id2item, file_path)

    def load_embedding(self, file_path):
        print("loading files from: ", file_path)
        user_emb_file = open(file_path + "_user.emb", "r")
        item_emb_file = open(file_path + "_item.emb", "r")

        user_emb_lines = user_emb_file.readlines()
        item_emb_lines = item_emb_file.readlines()

        n_user = int(user_emb_lines[0].split(" ")[0].strip())
        emb_dim = int(user_emb_lines[0].split(" ")[1].strip())
        n_item = int(item_emb_lines[0].split(" ")[0].strip())
        print("n_user:%d n_item:%d emb_dim:%d" % (n_user, n_item, emb_dim))
        user_emb = {}
        for i in range(1, n_user + 1):
            line_str = user_emb_lines[i].split(" ")
            user_id = line_str[0]
            emb_arr = line_str[1 : emb_dim + 1]
            user_emb[int(user_id)] = np.array(emb_arr, dtype=np.float)

        item_emb = {}
        for i in range(1, n_item + 1):
            line_str = item_emb_lines[i].split(" ")
            item_id = line_str[0]
            emb_arr = line_str[1 : emb_dim + 1]
            item_emb[int(item_id)] = np.array(emb_arr, dtype=np.float)

        return user_emb, item_emb

    def load_embedding_id(self, file_path):
        user_emb, item_emb = self.load_embedding(file_path)

        user_embedding = np.empty([self.data.n_users, self.emb_dim])
        for (user, u_id) in self.data.user2id.items():
            user_embedding[u_id] = user_emb[user]

        item_embedding = np.empty([self.data.n_items, self.emb_dim])
        for (item, i_id) in self.data.item2id.items():
            item_embedding[i_id] = item_emb[item]

        return user_embedding, item_embedding

    def predict(self, user_emb, item_emb, users, items):
        result = []
        for i in range(len(users)):
            s = np.dot(user_emb[users[i], :].squeeze(), item_emb[items[i], :].squeeze())
            result.append(s)
        return result

    def load_best_model(self):
        model = torch.load(self.output_file_name)
        self.model.load_state_dict(model)
        self.model.eval()
        return self.model
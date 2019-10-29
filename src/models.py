"""
Created on Aug 5, 2019
Update on XX,2019 BY xxx@

Classes describing datasets of user-item interactions. Instances of these
are returned by dataset fetching and dataset pre-processing functions.

@author: Zaiqiao Meng (zaiqiao.meng@gmail.com)

"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

torch.manual_seed(12345)

"""
Reproduce triple2ve by pytorch, where only one item embedding is used
"""
class Triple2vec_Single(nn.Module):
    def __init__(self, n_users, n_items, emb_dim, n_neg, batch_size):
        super(Triple2vec_Single, self).__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.emb_dim = emb_dim
        self.kl_loss = 0
        self.use_cuda = torch.cuda.is_available()
        self.n_neg = n_neg
        self.batch_size = batch_size
        self.user_emb = nn.Embedding(self.n_users, self.emb_dim)
        self.item_emb = nn.Embedding(self.n_items, self.emb_dim)
        self.user_bias = nn.Embedding(self.n_users, 1)
        self.item_bias = nn.Embedding(self.n_items, 1)

        if self.use_cuda:
            self.user_emb.cuda()
            self.item_emb.cuda()
            self.user_bias.cuda()
            self.item_bias.cuda()
        self.init_emb()

    """
    embedding initialization
    """
    def init_emb(self):
        initrange = 0.01  # 0.5 / self.emb_dim
        self.user_emb.weight.data.uniform_(-initrange, initrange)
        self.item_emb.weight.data.uniform_(-initrange, initrange)
        self.user_bias.weight.data.fill_(0.0)
        self.item_bias.weight.data.fill_(0.0)

    def forward(self, pos_u, pos_i_1, pos_i_2, neg_u, neg_i_1, neg_i_2):
        emb_u = self.user_emb(pos_u)
        emb_i_1 = self.item_emb(pos_i_1)
        emb_i_2 = self.item_emb(pos_i_2)

        emb_u_neg = self.user_emb(neg_u)
        emb_i_1_neg = self.item_emb(neg_i_2)
        emb_i_2_neg = self.item_emb(neg_i_2)

        input_emb_u = emb_i_1 + emb_i_2
        u_pos_score = torch.mul(emb_u, input_emb_u).squeeze()
        u_pos_score = torch.sum(u_pos_score, dim=1) + self.user_bias(pos_u).squeeze()
        u_pos_score = F.logsigmoid(u_pos_score)

        u_neg_score = (
            torch.bmm(emb_u_neg, emb_u.unsqueeze(2)).squeeze()
            + self.user_bias(neg_u).squeeze()
        )
        u_neg_score = F.logsigmoid(-1 * u_neg_score)
        u_score = -1 * (torch.sum(u_pos_score) + torch.sum(u_neg_score))

        input_emb_i_1 = emb_u + emb_i_2
        i_1_pos_score = torch.mul(emb_i_1, input_emb_i_1).squeeze()
        i_1_pos_score = (
            torch.sum(i_1_pos_score, dim=1) + self.item_bias(pos_i_1).squeeze()
        )
        i_1_pos_score = F.logsigmoid(i_1_pos_score)

        i_1_neg_score = (
            torch.bmm(emb_i_1_neg, emb_i_1.unsqueeze(2)).squeeze()
            + self.item_bias(neg_i_1).squeeze()
        )
        i_1_neg_score = F.logsigmoid(-1 * i_1_neg_score)

        i_1_score = -1 * (torch.sum(i_1_pos_score) + torch.sum(i_1_neg_score))

        input_emb_i_2 = emb_u + emb_i_1
        i_2_pos_score = torch.mul(emb_i_2, input_emb_i_2).squeeze()
        i_2_pos_score = (
            torch.sum(i_2_pos_score, dim=1) + self.item_bias(pos_i_2).squeeze()
        )
        i_2_pos_score = F.logsigmoid(i_2_pos_score)

        i_2_neg_score = (
            torch.bmm(emb_i_2_neg, emb_i_2.unsqueeze(2)).squeeze()
            + self.item_bias(neg_i_2).squeeze()
        )
        i_2_neg_score = F.logsigmoid(-1 * i_2_neg_score)

        i_2_score = -1 * (torch.sum(i_2_pos_score) + torch.sum(i_2_neg_score))

        return (u_score + i_1_score + i_2_score) / (3 * self.batch_size)

    def save_embedding(self, id2user, id2item, file_path):

        if self.use_cuda:
            self.u_embedding = self.user_emb.weight.cpu().data.numpy()
            self.i_embedding = self.item_emb.weight.cpu().data.numpy()
            self.u_bias = self.user_bias.weight.cpu().data.numpy()
            self.i_bias = self.item_bias.weight.cpu().data.numpy()
        else:
            self.u_embedding = self.user_emb.weight.data.numpy()
            self.i_embedding = self.item_emb.weight.data.numpy()
            self.u_bias = self.user_bias.weight.data.numpy()
            self.i_bias = self.item_bias.weight.data.numpy()

        fout = open(file_path + "_user.emb", "w")
        fout.write("%d %d\n" % (len(id2user), self.emb_dim))
        for wid, w in id2user.items():
            e = self.u_embedding[wid]
            e = " ".join(map(lambda x: str(x), e))
            fout.write("%s %s\n" % (w, e))

        fout = open(file_path + "_user.b", "w")
        fout.write("%d %d\n" % (len(id2user), 1))
        for wid, w in id2user.items():
            e = self.u_bias[wid]
            e = " ".join(map(lambda x: str(x), e))
            fout.write("%s %s\n" % (w, e))

        fout2 = open(file_path + "_item.emb", "w")
        fout2.write("%d %d\n" % (len(id2item), self.emb_dim))
        for wid, w in id2item.items():
            e = self.i_embedding[wid]
            e = " ".join(map(lambda x: str(x), e))
            fout2.write("%s %s\n" % (w, e))

        fout2 = open(file_path + "_item.b", "w")
        fout2.write("%d %d\n" % (len(id2item), 1))
        for wid, w in id2item.items():
            e = self.i_bias[wid]
            e = " ".join(map(lambda x: str(x), e))
            fout2.write("%s %s\n" % (w, e))

    def predict(self, users, items):
        result = []
        for i in range(len(users)):
            s = (
                np.dot(
                    self.u_embedding[users[i]].squeeze(),
                    self.i_embedding[items[i]].squeeze(),
                )
                + self.i_bias[items[i]].squeeze()
            )
            result.append(s)
        return result


"""
Reproduce triple2ve by pytorch in original setting, where only two item embeddings are used.
"""
class Triple2vec(nn.Module):
    def __init__(self, n_users, n_items, emb_dim, n_neg, batch_size):
        super(Triple2vec, self).__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.kl_loss = 0
        self.emb_dim = emb_dim
        self.use_cuda = torch.cuda.is_available()
        self.n_neg = n_neg
        self.batch_size = batch_size
        self.user_emb = nn.Embedding(self.n_users, self.emb_dim)
        self.item_emb = nn.Embedding(self.n_items, self.emb_dim)
        self.item_emb2 = nn.Embedding(self.n_items, self.emb_dim)
        self.user_bias = nn.Embedding(self.n_users, 1)
        self.item_bias = nn.Embedding(self.n_items, 1)

        if self.use_cuda:
            self.user_emb.cuda()
            self.item_emb.cuda()
            self.item_emb2.cuda()
            self.user_bias.cuda()
            self.item_bias.cuda()
        self.init_emb()

    def init_emb(self):
        #         initrange = 0.5 / self.emb_dim
        self.user_emb.weight.data.uniform_(-0.01, 0.01)
        self.item_emb.weight.data.uniform_(-0.01, 0.01)
        self.item_emb2.weight.data.uniform_(-0.01, 0.01)
        self.user_bias.weight.data.fill_(0.0)
        self.item_bias.weight.data.fill_(0.0)

    def forward(self, pos_u, pos_i_1, pos_i_2, neg_u, neg_i_1, neg_i_2):
        emb_u = self.user_emb(pos_u)
        emb_i_1 = self.item_emb(pos_i_1)
        emb_i_2 = self.item_emb2(pos_i_2)

        emb_u_neg = self.user_emb(neg_u)
        emb_i_1_neg = self.item_emb(neg_i_2)
        emb_i_2_neg = self.item_emb2(neg_i_2)

        input_emb_u = emb_i_1 + emb_i_2
        u_pos_score = torch.mul(emb_u, input_emb_u).squeeze()
        u_pos_score = torch.sum(u_pos_score, dim=1) + self.user_bias(pos_u).squeeze()
        u_pos_score = F.logsigmoid(u_pos_score)

        u_neg_score = (
            torch.bmm(emb_u_neg, emb_u.unsqueeze(2)).squeeze()
            + self.user_bias(neg_u).squeeze()
        )
        u_neg_score = F.logsigmoid(-1 * u_neg_score)
        u_score = -1 * (torch.sum(u_pos_score) + torch.sum(u_neg_score))

        input_emb_i_1 = emb_u + emb_i_2
        i_1_pos_score = torch.mul(emb_i_1, input_emb_i_1).squeeze()
        i_1_pos_score = (
            torch.sum(i_1_pos_score, dim=1) + self.item_bias(pos_i_1).squeeze()
        )
        i_1_pos_score = F.logsigmoid(i_1_pos_score)

        i_1_neg_score = (
            torch.bmm(emb_i_1_neg, emb_i_1.unsqueeze(2)).squeeze()
            + self.item_bias(neg_i_1).squeeze()
        )
        i_1_neg_score = F.logsigmoid(-1 * i_1_neg_score)

        i_1_score = -1 * (torch.sum(i_1_pos_score) + torch.sum(i_1_neg_score))

        input_emb_i_2 = emb_u + emb_i_1
        i_2_pos_score = torch.mul(emb_i_2, input_emb_i_2).squeeze()
        i_2_pos_score = (
            torch.sum(i_2_pos_score, dim=1) + self.item_bias(pos_i_2).squeeze()
        )
        i_2_pos_score = F.logsigmoid(i_2_pos_score)

        i_2_neg_score = (
            torch.bmm(emb_i_2_neg, emb_i_2.unsqueeze(2)).squeeze()
            + self.item_bias(neg_i_2).squeeze()
        )
        i_2_neg_score = F.logsigmoid(-1 * i_2_neg_score)

        i_2_score = -1 * (torch.sum(i_2_pos_score) + torch.sum(i_2_neg_score))

        return (u_score + i_1_score + i_2_score) / (3 * self.batch_size)

    def save_embedding(self, id2user, id2item, file_path):

        if self.use_cuda:
            self.u_embedding = self.user_emb.weight.cpu().data.numpy()
            self.i_embedding = (
                self.item_emb.weight.cpu().data.numpy()
                + self.item_emb2.weight.cpu().data.numpy()
            ) / 2
            self.u_bias = self.user_bias.weight.cpu().data.numpy()
            self.i_bias = self.item_bias.weight.cpu().data.numpy()
        else:
            self.u_embedding = self.user_emb.weight.data.numpy()
            self.i_embedding = (
                self.item_emb.weight.data.numpy() + self.item_emb2.weight.data.numpy()
            ) / 2
            self.u_bias = self.user_bias.weight.data.numpy()
            self.i_bias = self.item_bias.weight.data.numpy()

        fout = open(file_path + "_user.emb", "w")
        fout.write("%d %d\n" % (len(id2user), self.emb_dim))
        for wid, w in id2user.items():
            e = self.u_embedding[wid]
            e = " ".join(map(lambda x: str(x), e))
            fout.write("%s %s\n" % (w, e))

        fout = open(file_path + "_user.b", "w")
        fout.write("%d %d\n" % (len(id2user), 1))
        for wid, w in id2user.items():
            e = self.u_bias[wid]
            e = " ".join(map(lambda x: str(x), e))
            fout.write("%s %s\n" % (w, e))

        fout2 = open(file_path + "_item.emb", "w")
        fout2.write("%d %d\n" % (len(id2item), self.emb_dim))
        for wid, w in id2item.items():
            e = self.i_embedding[wid]
            e = " ".join(map(lambda x: str(x), e))
            fout2.write("%s %s\n" % (w, e))

        fout2 = open(file_path + "_item.b", "w")
        fout2.write("%d %d\n" % (len(id2item), 1))
        for wid, w in id2item.items():
            e = self.i_bias[wid]
            e = " ".join(map(lambda x: str(x), e))
            fout2.write("%s %s\n" % (w, e))

    def predict(self, users, items):
        if self.use_cuda:
            self.u_embedding = self.user_emb.weight.cpu().data.numpy()
            self.i_embedding = (
                self.item_emb.weight.cpu().data.numpy()
                + self.item_emb2.weight.cpu().data.numpy()
            ) / 2
            self.u_bias = self.user_bias.weight.cpu().data.numpy()
            self.i_bias = self.item_bias.weight.cpu().data.numpy()
        else:
            self.u_embedding = self.user_emb.weight.data.numpy()
            self.i_embedding = (
                self.item_emb.weight.data.numpy() + self.item_emb2.weight.data.numpy()
            ) / 2
            self.u_bias = self.user_bias.weight.data.numpy()
            self.i_bias = self.item_bias.weight.data.numpy()
        result = []
        for i in range(len(users)):
            s = (
                np.dot(
                    self.u_embedding[users[i]].squeeze(),
                    self.i_embedding[items[i]].squeeze(),
                )
                + self.i_bias[items[i]].squeeze()
            )
            result.append(s)
        return result

    
"""
Our VBCAR model
"""

class VAE(nn.Module):
    def __init__(
        self,
        n_users,
        n_items,
        user_fea,
        item_fea,
        late_dim,
        emb_dim,
        neg_n,
        batch_size,
        activator="tanh",
        alpha=0.0,
        device=torch.device("cpu"),
    ):
        super(VAE, self).__init__()
        self.device = device
        self.n_users = n_users
        self.n_items = n_items
        self.user_fea = user_fea
        self.item_fea = item_fea
        self.late_dim = late_dim
        self.emb_dim = emb_dim
        self.neg_n = neg_n
        self.batch_size = batch_size
        if activator == "tanh":
            self.act = torch.tanh
        elif activator == "sigmoid":
            self.act = torch.sigmoid
        elif activator == "relu":
            self.act = torch.relu
        else:
            self.act = torch.tanh

        self.alpha = alpha
        self.user_fea_dim = self.user_fea.shape[1]
        self.item_fea_dim = self.item_fea.shape[1]
        print(self.user_fea_dim, self.item_fea_dim)
        self.init_layers()

    def init_layers(self):
        self.fc_u_1 = nn.Linear(self.user_fea_dim, self.late_dim).to(self.device)
        self.fc_u_2 = nn.Linear(self.late_dim, self.emb_dim * 2).to(self.device)
        self.fc_i_1 = nn.Linear(self.item_fea_dim, self.late_dim).to(self.device)
        self.fc_i_2 = nn.Linear(self.late_dim, self.emb_dim * 2).to(self.device)

    def user_encode(self, index):
        x = self.user_fea[index]
        h1 = self.act(self.fc_u_1(x))
        h2 = self.fc_u_2(h1)
        return h2

    def item_encode(self, index):
        x = self.item_fea[index]
        h1 = self.act(self.fc_i_1(x))
        h2 = self.fc_i_2(h1)
        return h2

    def reparameterize(self, hidden):
        mu = hidden[..., : self.emb_dim]
        logvar = hidden[..., self.emb_dim :]
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, pos_u, pos_i_1, pos_i_2, neg_u, neg_i_1, neg_i_2):
        pos_u_dis = self.user_encode(pos_u)
        pos_u_emb = self.reparameterize(pos_u_dis)
        #         pos_u_bias = pos_u_dis[...,self.emb_dim*2]

        pos_i_1_dis = self.item_encode(pos_i_1)
        pos_i_1_emb = self.reparameterize(pos_i_1_dis)
        #         pos_i_1_bias = pos_i_1_dis[...,self.emb_dim*2]

        pos_i_2_dis = self.item_encode(pos_i_2)
        pos_i_2_emb = self.reparameterize(pos_i_2_dis)
        #         pos_i_2_bias = pos_i_2_dis[...,self.emb_dim*2]

        neg_u_dis = self.user_encode(neg_u.view(-1))
        neg_u_emb = self.reparameterize(neg_u_dis).view(-1, self.neg_n, self.emb_dim)
        #         neg_u_bias = neg_u_dis[...,self.emb_dim*2].view(-1,self.neg_n)

        neg_i_1_dis = self.item_encode(neg_i_1.view(-1))
        neg_i_1_emb = self.reparameterize(neg_i_1_dis).view(
            -1, self.neg_n, self.emb_dim
        )
        #         neg_i_1_bias = neg_i_1_dis[...,self.emb_dim*2].view(-1,self.neg_n)

        neg_i_2_dis = self.item_encode(neg_i_2.view(-1))
        neg_i_2_emb = self.reparameterize(neg_i_2_dis).view(
            -1, self.neg_n, self.emb_dim
        )
        #         neg_i_2_bias = neg_i_2_dis[...,self.emb_dim*2].view(-1,self.neg_n)

        input_emb_u = pos_i_1_emb + pos_i_2_emb

        u_pos_score = torch.mul(pos_u_emb, input_emb_u).squeeze()
        u_pos_score = torch.sum(u_pos_score, dim=1)  # + pos_u_bias.squeeze()
        u_pos_score = F.logsigmoid(u_pos_score)

        # neg_u_emb 128*5*16 pos_u_emb 128*16 pos_u_emb.unsqueeze(2) 128*16*1
        # torch.bmm(neg_u_emb, pos_u_emb.unsqueeze(2)) 128*5*1
        u_neg_score = torch.bmm(
            neg_u_emb, pos_u_emb.unsqueeze(2)
        ).squeeze()  # +neg_u_bias.squeeze()
        u_neg_score = F.logsigmoid(-1 * u_neg_score)
        u_score = -1 * (torch.sum(u_pos_score) + torch.sum(u_neg_score))

        input_emb_i_1 = pos_u_emb + pos_i_2_emb
        i_1_pos_score = torch.mul(pos_i_1_emb, input_emb_i_1).squeeze()
        i_1_pos_score = torch.sum(i_1_pos_score, dim=1)  # + pos_i_1_bias.squeeze()
        i_1_pos_score = F.logsigmoid(i_1_pos_score)
        i_1_neg_score = torch.bmm(
            neg_i_1_emb, pos_i_1_emb.unsqueeze(2)
        ).squeeze()  # +neg_i_1_bias.squeeze()
        i_1_neg_score = F.logsigmoid(-1 * i_1_neg_score)
        i_1_score = -1 * (torch.sum(i_1_pos_score) + torch.sum(i_1_neg_score))

        input_emb_i_2 = pos_u_emb + pos_i_1_emb
        i_2_pos_score = torch.mul(pos_i_2_emb, input_emb_i_2).squeeze()
        i_2_pos_score = torch.sum(i_2_pos_score, dim=1)  # + pos_i_2_bias.squeeze()
        i_2_pos_score = F.logsigmoid(i_2_pos_score)

        i_2_neg_score = torch.bmm(
            neg_i_2_emb, pos_i_2_emb.unsqueeze(2)
        ).squeeze()  # +neg_i_2_bias.squeeze()
        i_2_neg_score = F.logsigmoid(-1 * i_2_neg_score)
        i_2_score = -1 * (torch.sum(i_2_pos_score) + torch.sum(i_2_neg_score))

        cum_mu = torch.cat(
            (
                pos_u_dis[..., : self.emb_dim],
                pos_i_1_dis[..., : self.emb_dim],
                pos_i_2_dis[..., : self.emb_dim],
                neg_u_dis[..., : self.emb_dim],
                neg_i_1_dis[..., : self.emb_dim],
                neg_i_2_dis[..., : self.emb_dim],
            ),
            0,
        )

        cum_logvar = torch.cat(
            (
                pos_u_dis[..., self.emb_dim : self.emb_dim * 2],
                pos_i_1_dis[..., self.emb_dim : self.emb_dim * 2],
                pos_i_2_dis[..., self.emb_dim : self.emb_dim * 2],
                neg_u_dis[..., self.emb_dim : self.emb_dim * 2],
                neg_i_1_dis[..., self.emb_dim : self.emb_dim * 2],
                neg_i_2_dis[..., self.emb_dim : self.emb_dim * 2],
            ),
            0,
        )
        KLD = (
            -0.5
            * torch.mean(1 + cum_logvar - cum_mu.pow(2) - cum_logvar.exp())
        )
        self.kl_loss = KLD.clone()
        ### to do: normalization
        return (
            (1 - self.alpha) * (u_score + i_1_score + i_2_score) / (3 * self.batch_size)
        ) + (self.alpha * KLD)

    def save_embedding(self, id2user, id2item, file_path):
        all_users = torch.tensor(
            np.arange(self.n_users), dtype=torch.int64, device=self.device
        )
        all_items = torch.tensor(
            np.arange(self.n_items), dtype=torch.int64, device=self.device
        )

        if self.device.type == "cuda":
            self.u_embedding = self.user_encode(all_users).cpu().detach().numpy()
            self.i_embedding = self.item_encode(all_items).cpu().detach().numpy()
        else:
            self.u_embedding = self.user_encode(all_users).detach().numpy()
            self.i_embedding = self.item_encode(all_items).detach().numpy()

        fout = open(file_path + "_user.emb", "w")
        fout.write("%d %d\n" % (len(id2user), self.emb_dim))
        for wid, w in id2user.items():
            e = self.u_embedding[wid]
            e = " ".join(map(lambda x: str(x), e))
            fout.write("%s %s\n" % (w, e))

        fout2 = open(file_path + "_item.emb", "w")
        fout2.write("%d %d\n" % (len(id2item), self.emb_dim))
        for wid, w in id2item.items():
            e = self.i_embedding[wid]
            e = " ".join(map(lambda x: str(x), e))
            fout2.write("%s %s\n" % (w, e))

    def predict(self, users, items):
        all_users = torch.tensor(
            np.arange(self.n_users), dtype=torch.int64, device=self.device
        )
        all_items = torch.tensor(
            np.arange(self.n_items), dtype=torch.int64, device=self.device
        )

        if self.device.type == "cuda":
            self.u_embedding = self.user_encode(all_users).cpu().detach().numpy()
            self.i_embedding = self.item_encode(all_items).cpu().detach().numpy()
        else:
            self.u_embedding = self.user_encode(all_users).detach().numpy()
            self.i_embedding = self.item_encode(all_items).detach().numpy()
        result = []
        for i in range(len(users)):
            s = np.dot(
                self.u_embedding[users[i], : self.emb_dim].squeeze(),
                self.i_embedding[items[i], : self.emb_dim].squeeze(),
            )  # +self.i_embedding[items[i],self.emb_dim].squeeze()
            result.append(s)
        return result
import math
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.init import xavier_normal_, xavier_uniform_
import gc
        
class TransE(nn.Module):
    def __init__(self, num_ent, num_rel, emb_dim, device, margin=2.0):
        super(TransE, self).__init__()
        self.num_ent = num_ent
        self.num_rel = num_rel
        self.emb_dim = emb_dim
        self.device = device
        self.distance = 'l2'
        self.margin = margin
        self.gamma = margin
        self.epsilon = 2.0
        self.p_norm = 1
        self.ent_embs = nn.Embedding(self.num_ent, self.emb_dim).to(self.device)
        self.rel_embs = nn.Embedding(self.num_rel, self.emb_dim).to(self.device)
        self.margin = nn.Parameter(
            torch.Tensor([self.gamma]).to(self.device),
            requires_grad=False
        )
        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.margin.item() + self.epsilon) / self.emb_dim]).to(self.device),
            requires_grad=False
        )
        nn.init.uniform_(tensor=self.ent_embs.weight.data, a=-self.embedding_range.item(), b=self.embedding_range.item())
        nn.init.uniform_(tensor=self.rel_embs.weight.data, a=-self.embedding_range.item(), b=self.embedding_range.item())   

    def forward(self, hs, rs, ts):
        e_hs, e_rs, e_ts = self.ent_embs(hs).to(self.device), self.rel_embs(rs).to(self.device), self.ent_embs(ts).to(self.device)
        score = self._calc(e_hs, e_rs, e_ts).view(-1, 1).to(self.device)
        return score

    def _calc(self, e_hs, e_rs, e_ts):
        score = (e_hs + e_rs) - e_ts
        if self.distance == 'l1':
            score = torch.norm(score, self.p_norm, -1)
        else:
            score = torch.sqrt(torch.sum((score)**2, 1))
        score = score - self.margin
        return -score
    
    def _loss(self, y_pos, y_neg, neg_ratio):
        P = y_pos.size(0)
        y_pos = y_pos.view(-1).repeat(int(y_neg.shape[0]/P)).to(self.device)
        y_neg = y_neg.view(-1).to(self.device)
        target = Variable(torch.from_numpy(np.ones(P*(int(y_neg.shape[0]/P)), dtype=np.int32))).to(self.device)
        criterion = nn.MarginRankingLoss(margin=self.gamma)
        loss = criterion(y_pos, y_neg, target)
        return loss

    def _regularization(self, hs, rs, ts):
        e_hs, e_rs, e_ts = self.ent_embs(hs).to(self.device), self.rel_embs(rs).to(self.device), self.ent_embs(ts).to(self.device)
        regul = (torch.mean(e_hs ** 2) + torch.mean(e_ts ** 2) + torch.mean(e_rs ** 2)) / 3
        return regul

    def predict(self, hs, rs, ts):
        y_pred = self.forward(hs, rs, ts).view(-1, 1)
        return y_pred.data


class TransH(nn.Module):
    def __init__(self, num_ent, num_rel, emb_dim, device, margin=2.0, norm_flag=True):
        super(TransH, self).__init__()
        self.num_ent = num_ent
        self.num_rel = num_rel
        self.emb_dim = emb_dim
        self.device = device

        self.distance = 'l2'
        self.gamma = margin
        self.margin = margin
        self.epsilon = 2.0
        self.norm = 1

        self.margin = nn.Parameter(
            torch.Tensor([self.gamma]).to(self.device),
            requires_grad=False
        )
        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.margin.item() + self.epsilon) / self.emb_dim]).to(self.device),
            requires_grad=False
        )

        self.ent_embs   = nn.Embedding(num_ent, emb_dim).to(device)
        self.rel_embs   = nn.Embedding(num_rel, emb_dim).to(device)
        self.norm_vector = nn.Embedding(self.num_rel, self.emb_dim).to(self.device)
        
        nn.init.uniform_(
            tensor = self.ent_embs.weight.data, 
            a = -self.embedding_range.item(), 
            b = self.embedding_range.item()
        )
        nn.init.uniform_(
            tensor = self.rel_embs.weight.data, 
            a= -self.embedding_range.item(), 
            b= self.embedding_range.item()
        )
        nn.init.uniform_(
            tensor = self.norm_vector.weight.data, 
            a= -self.embedding_range.item(), 
            b= self.embedding_range.item()
        )
        
    def forward(self, hs, rs, ts):
        e_hs, e_rs, e_ts = self.ent_embs(hs).to(self.device), self.rel_embs(rs).to(self.device), self.ent_embs(ts).to(self.device)
        n = self.norm_vector(rs).to(self.device)
        proj_e_hs, proj_e_ts = self._transfer(e_hs, n).to(self.device), self._transfer(e_ts, n).to(self.device)
        score = self._calc(proj_e_hs, e_rs, proj_e_ts).view(-1, 1).to(self.device)
        return score
        
    def _transfer(self,embeddings,norm):
        return embeddings - torch.sum(embeddings * norm, 1, True) * norm

    def _calc(self, e_hs, e_rs, e_ts):
        score = (e_hs + e_rs) - e_ts
        if self.distance == 'l1':
            score = torch.norm(score, self.p_norm, -1)
        else:
            score = torch.sqrt(torch.sum((score)**2, 1))
        score = score - self.margin
        return -score

    def _loss(self, y_pos, y_neg, neg_ratio):
        P = y_pos.size(0)
        y_pos = y_pos.view(-1).repeat(int(y_neg.shape[0]/P)).to(self.device)
        y_neg = y_neg.view(-1).to(self.device)
        target = Variable(torch.from_numpy(np.ones(P*(int(y_neg.shape[0]/P)), dtype=np.int32))).to(self.device)
        criterion = nn.MarginRankingLoss(margin=self.gamma)
        loss = criterion(y_pos, y_neg, target)
        return loss

    def _regularization(self, hs, rs, ts):
        e_hs, e_rs, e_ts = self.ent_embs(hs).to(self.device), self.rel_embs(rs).to(self.device), self.ent_embs(ts).to(self.device)
        r_norm = self.norm_vector(rs)
        regul = (torch.mean(e_hs ** 2) + 
                 torch.mean(e_rs ** 2) + 
                 torch.mean(e_ts ** 2) +
                 torch.mean(r_norm ** 2)) / 4
        return regul

    def predict(self, hs, rs, ts):
        y_pred = self.forward(hs, rs, ts).view(-1, 1)
        return y_pred.data

class TransR(nn.Module):
    def __init__(self, num_ent, num_rel, emb_dim, device, margin=2.0, norm_flag=True, rand_init = True):
        super(TransR, self).__init__()
        self.dim_e = emb_dim
        self.dim_r = emb_dim
        self.norm_flag = norm_flag
        self.p_norm = 1
        self.rand_init = rand_init
        self.device = device
        self.num_ent = num_ent
        self.num_rel = num_rel

        self.ent_embs = nn.Embedding(self.num_ent, self.dim_e).to(self.device)
        self.rel_embs = nn.Embedding(self.num_rel, self.dim_r).to(self.device)
        nn.init.xavier_uniform_(self.ent_embs.weight.data)
        nn.init.xavier_uniform_(self.rel_embs.weight.data)
        self.transfer_matrix = nn.Embedding(self.num_rel, self.dim_e * self.dim_r).to(self.device)

        if not self.rand_init:
            identity = torch.zeros(self.dim_e, self.dim_r).to(self.device)
            for i in range(min(self.dim_e, self.dim_r)):
                identity[i][i] = 1
            identity = identity.view(self.dim_r * self.dim_e).to(self.device)
            for i in range(self.num_rel):
                self.transfer_matrix.weight.data[i] = identity
        else:
            nn.init.xavier_uniform_(self.transfer_matrix.weight.data)

        if margin != None:
            self.margin = nn.Parameter(torch.Tensor([margin]).to(self.device))
            self.margin.requires_grad = False
            self.margin_flag = True
        else:
            self.margin_flag = False

    def _calc(self, e_hs, e_rs, e_ts):
        score = (e_hs + e_rs) - e_ts
        if self.p_norm == 1:
            score = torch.norm(score, self.p_norm, -1)
        else:
            score = torch.sqrt(torch.sum((score)**2, 1))
        #TEST
        if self.margin_flag == True:
            score = score - self.margin
        return -score
    
    def _transfer(self, e, r_transfer):
        r_transfer = r_transfer.view(-1, self.dim_e, self.dim_r)
        if e.shape[0] != r_transfer.shape[0]:
            e = e.view(-1, r_transfer.shape[0], self.dim_e).permute(1, 0, 2)
            e = torch.matmul(e, r_transfer).permute(1, 0, 2)
        else:
            e = e.view(-1, 1, self.dim_e)
            e = torch.matmul(e, r_transfer)
        return e.view(-1, self.dim_r)

    def forward(self, hs, rs, ts):
        torch.cuda.empty_cache()
        gc.collect()
        h = self.ent_embs(hs).to(self.device)
        t = self.ent_embs(ts).to(self.device)
        r = self.rel_embs(rs).to(self.device)
        r_transfer = self.transfer_matrix(rs)
        h = self._transfer(h, r_transfer)
        t = self._transfer(t, r_transfer)
        score = self._calc(h ,t, r).to(self.device)
        del h
        del t
        torch.cuda.empty_cache()
        gc.collect()
        return score

    def _regularization(self, hs, rs, ts):
        h = self.ent_embs(hs).to(self.device)
        t = self.ent_embs(ts).to(self.device)
        r = self.rel_embs(rs).to(self.device)
        r_transfer = self.transfer_matrix(rs)
        regul = (torch.mean(h ** 2) + 
                 torch.mean(t ** 2) + 
                 torch.mean(r ** 2) +
                 torch.mean(r_transfer ** 2)) / 4
        return regul * regul

    def _loss(self, y_pos, y_neg, neg_ratio):
        P = y_pos.size(0)
        y_pos = y_pos.view(-1).repeat(int(y_neg.shape[0]/P)).to(self.device)
        y_neg = y_neg.view(-1).to(self.device)
        target = Variable(torch.from_numpy(np.ones(P*(int(y_neg.shape[0]/P)), dtype=np.int32))).to(self.device)
        criterion = nn.MarginRankingLoss(margin=2.0)
        loss = criterion(y_pos, y_neg, target)
        return loss

class TransD(nn.Module):
    def __init__(self, num_ent, num_rel, emb_dim, device, margin=2.0, norm_flag=True):
        super(TransD, self).__init__()
        self.num_ent = num_ent
        self.num_rel = num_rel
        self.dim_e = emb_dim
        self.dim_r = emb_dim
        self.device = device

        self.distance = 'l2'
        self.margin = margin
        self.epsilon = 2.0
        self.norm = 1

        self.ent_embs   = nn.Embedding(self.num_ent, self.dim_e).to(self.device)
        self.rel_embs   = nn.Embedding(self.num_rel, self.dim_r).to(self.device)
        self.ent_transfer = nn.Embedding(self.num_ent, self.dim_e).to(self.device)
        self.rel_transfer = nn.Embedding(self.num_rel, self.dim_r).to(self.device)

        if self.margin == None or self.epsilon == None:
            nn.init.xavier_uniform_(self.ent_embs.weight.data)
            nn.init.xavier_uniform_(self.rel_embs.weight.data)
            nn.init.xavier_uniform_(self.ent_transfer.weight.data)
            nn.init.xavier_uniform_(self.rel_transfer.weight.data)
        else:
            self.ent_embedding_range = nn.Parameter(
                torch.Tensor([(self.margin + self.epsilon) / self.dim_e]).to(self.device), requires_grad=False
            )
            self.rel_embedding_range = nn.Parameter(
                torch.Tensor([(self.margin + self.epsilon) / self.dim_r]).to(self.device), requires_grad=False
            )
            nn.init.uniform_(
                tensor = self.ent_embs.weight.data, 
                a = -self.ent_embedding_range.item(), 
                b = self.ent_embedding_range.item()
            )
            nn.init.uniform_(
                tensor = self.rel_embs.weight.data, 
                a= -self.rel_embedding_range.item(), 
                b= self.rel_embedding_range.item()
            )
            nn.init.uniform_(
                tensor = self.ent_transfer.weight.data, 
                a= -self.ent_embedding_range.item(), 
                b= self.ent_embedding_range.item()
            )
            nn.init.uniform_(
                tensor = self.rel_transfer.weight.data, 
                a= -self.rel_embedding_range.item(), 
                b= self.rel_embedding_range.item()
            )
        if margin != None:
            self.margin = nn.Parameter(torch.Tensor([margin]).to(self.device))
            self.margin.requires_grad = False
            self.margin_flag = True
        else:
            self.margin_flag = False

    def _resize(self, tensor, axis, size):
        shape = tensor.size()
        osize = shape[axis]
        if osize == size:
            return tensor
        if (osize > size):
            return torch.narrow(tensor, axis, 0, size)
        paddings = []
        for i in range(len(shape)):
            if i == axis:
                paddings = [0, size - osize] + paddings
            else:
                paddings = [0, 0] + paddings
        print (paddings)
        return F.pad(tensor, paddings = paddings, mode = "constant", value = 0)

    def _transfer(self, e, e_transfer, r_transfer):
        if e.shape[0] != r_transfer.shape[0]:
            e = e.view(-1, r_transfer.shape[0], e.shape[-1])
            e_transfer = e_transfer.view(-1, r_transfer.shape[0], e_transfer.shape[-1])
            r_transfer = r_transfer.view(-1, r_transfer.shape[0], r_transfer.shape[-1])
            e = F.normalize(
                self._resize(e, -1, r_transfer.size()[-1]) + torch.sum(e * e_transfer, -1, True) * r_transfer,
                p = 2, 
                dim = -1
            )           
            return e.view(-1, e.shape[-1]).to(self.device)
        else:
            return F.normalize(
                self._resize(e, -1, r_transfer.size()[-1]) + torch.sum(e * e_transfer, -1, True) * r_transfer,
                p = 2, 
                dim = -1
            ).to(self.device)

    def forward(self, hs, rs, ts):
        h = self.ent_embs(hs).to(self.device)
        t = self.ent_embs(ts).to(self.device)
        r = self.rel_embs(rs).to(self.device)
        h_transfer = self.ent_transfer(hs).to(self.device)
        t_transfer = self.ent_transfer(ts).to(self.device)
        r_transfer = self.rel_transfer(rs).to(self.device)
        h = self._transfer(h, h_transfer, r_transfer)
        t = self._transfer(t, t_transfer, r_transfer)
        score = self._calc(h ,r, t).to(self.device)
        return score

    def _calc(self, e_hs, e_rs, e_ts):
        score = (e_hs + e_rs) - e_ts
        if self.norm == 1:
            score = torch.norm(score, self.norm, -1)
        else:
            score = torch.sqrt(torch.sum((score)**2, 1))
        #TEST
        if self.margin_flag == True:
            score = score - self.margin
        return -score

    def _regularization(self, hs, rs, ts):
        h = self.ent_embs(hs).to(self.device)
        t = self.ent_embs(ts).to(self.device)
        r = self.rel_embs(rs).to(self.device)
        h_transfer = self.ent_transfer(hs).to(self.device)
        t_transfer = self.ent_transfer(ts).to(self.device)
        r_transfer = self.rel_transfer(rs).to(self.device)
        regul = (torch.mean(h ** 2) + 
                 torch.mean(t ** 2) + 
                 torch.mean(r ** 2) + 
                 torch.mean(h_transfer ** 2) + 
                 torch.mean(t_transfer ** 2) + 
                 torch.mean(r_transfer ** 2)) / 6
        return regul

    def _loss(self, y_pos, y_neg, neg_ratio):
        P = y_pos.size(0)
        y_pos = y_pos.view(-1).repeat(int(y_neg.shape[0]/P)).to(self.device)
        y_neg = y_neg.view(-1).to(self.device)
        target = Variable(torch.from_numpy(np.ones(P*(int(y_neg.shape[0]/P)), dtype=np.int32))).to(self.device)
        criterion = nn.MarginRankingLoss(margin=2.0)
        loss = criterion(y_pos, y_neg, target)
        return loss

### SEMANTIC-MATCHING MODELS ###

class DistMult(nn.Module):
    def __init__(self, num_ent, num_rel, emb_dim, device, margin=5.0):
        super(DistMult, self).__init__()
        self.num_ent = num_ent
        self.num_rel = num_rel
        self.emb_dim = emb_dim
        self.device = device
        self.distance = 'l2'
        self.gamma = margin
        self.ent_embs   = nn.Embedding(num_ent, emb_dim).to(device)
        self.rel_embs   = nn.Embedding(num_rel, emb_dim).to(device)
        nn.init.xavier_uniform_(self.ent_embs.weight.data)
        nn.init.xavier_uniform_(self.rel_embs.weight.data)

    def forward(self, hs, rs, ts):
        e_hs, e_rs, e_ts = self.ent_embs(hs).to(self.device), self.rel_embs(rs).to(self.device), self.ent_embs(ts).to(self.device)
        score = self._calc(e_hs, e_rs, e_ts) 
        return score

    def _calc(self,e_hs,e_rs,e_ts):
        return torch.sum(e_hs*e_rs*e_ts,-1)
    
    def _loss(self, y_pos, y_neg, neg_ratio):
        P = y_pos.size(0)
        y_pos = y_pos.view(-1).repeat(int(y_neg.shape[0]/P)).to(self.device)
        y_neg = y_neg.view(-1).to(self.device)
        target = Variable(torch.from_numpy(np.ones(P*(int(y_neg.shape[0]/P)), dtype=np.int32))).to(self.device)
        criterion = nn.MarginRankingLoss(margin=self.gamma)
        loss = criterion(y_pos, y_neg, target)
        return loss

    def _regularization(self, hs, rs, ts):
        e_hs, e_rs, e_ts = self.ent_embs(hs).to(self.device), self.rel_embs(rs).to(self.device), self.ent_embs(ts).to(self.device)
        regul = (torch.mean(e_hs ** 2) + torch.mean(e_ts ** 2) + torch.mean(e_rs ** 2))/3
        return regul


class ComplEx(nn.Module):
    def __init__(self, num_ent, num_rel, emb_dim, device, margin=5.0, lmbda=0.0):
        super(ComplEx,self).__init__()
        self.num_ent = num_ent
        self.num_rel = num_rel
        self.emb_dim = emb_dim
        self.gamma = margin
        self.lmbda = lmbda
        self.device = device
        self.ent_re_embeddings=nn.Embedding(self.num_ent, self.emb_dim).to(self.device)
        self.ent_im_embeddings=nn.Embedding(self.num_ent, self.emb_dim).to(self.device)
        self.rel_re_embeddings=nn.Embedding(self.num_rel, self.emb_dim).to(self.device)
        self.rel_im_embeddings=nn.Embedding(self.num_rel, self.emb_dim).to(self.device)
        self.criterion = nn.Softplus()
        self.init_weights()
        
    def init_weights(self):
        nn.init.xavier_uniform_(self.ent_re_embeddings.weight.data)
        nn.init.xavier_uniform_(self.ent_im_embeddings.weight.data)
        nn.init.xavier_uniform_(self.rel_re_embeddings.weight.data)
        nn.init.xavier_uniform_(self.rel_im_embeddings.weight.data)
        
    def _calc(self,e_re_h,e_im_h,r_re,r_im,e_re_t,e_im_t):
        return torch.sum(r_re*e_re_h*e_re_t + r_re*e_im_h*e_im_t + r_im*e_re_h*e_im_t - r_im*e_im_h*e_re_t,1,False)

    def _loss(self, pos_scores, neg_scores, neg_ratio):
        P, N = pos_scores.size(0), neg_scores.size(0)
        pos_scores, neg_scores = pos_scores.view(-1).to(self.device), neg_scores.view(-1).to(self.device)
        true_y, corrup_y = torch.ones(P).to(self.device), -torch.ones(N).to(self.device)
        target = torch.cat((true_y, corrup_y), 0)
        y = torch.cat((pos_scores, neg_scores), 0).to(self.device)
        return torch.mean(self.criterion(-target * y))
    
    def forward(self, hs, rs, ts):
        e_re_h, e_im_h = self.ent_re_embeddings(hs).to(self.device), self.ent_im_embeddings(hs).to(self.device)
        e_re_t, e_im_t = self.ent_re_embeddings(ts).to(self.device), self.ent_im_embeddings(ts).to(self.device)
        r_re, r_im = self.rel_re_embeddings(rs).to(self.device), self.rel_im_embeddings(rs).to(self.device)
        score = self._calc(e_re_h,e_im_h,r_re,r_im,e_re_t,e_im_t)
        return score 

    def _regularization(self, hs, rs, ts):
        e_re_h, e_im_h = self.ent_re_embeddings(hs).to(self.device), self.ent_im_embeddings(hs).to(self.device)
        e_re_t, e_im_t = self.ent_re_embeddings(ts).to(self.device), self.ent_im_embeddings(ts).to(self.device)
        r_re, r_im = self.rel_re_embeddings(rs).to(self.device), self.rel_im_embeddings(rs).to(self.device)
        regul = (torch.mean(e_re_h ** 2) + 
                 torch.mean(e_im_h ** 2) + 
                 torch.mean(e_re_t ** 2) +
                 torch.mean(e_im_t ** 2) +
                 torch.mean(r_re ** 2) +
                 torch.mean(r_im ** 2)) / 6
        return regul


class SimplE2(nn.Module):
    def __init__(self, num_ent, num_rel, emb_dim, device):
        super(SimplE2, self).__init__()
        self.dim = emb_dim
        self.num_ent = num_ent
        self.num_rel = num_rel
        self.device = device
        self.criterion = nn.Softplus()
        self.ent_h_embs   = nn.Embedding(num_ent, self.dim).to(self.device)
        self.ent_t_embs   = nn.Embedding(num_ent, self.dim).to(self.device)
        self.rel_embs     = nn.Embedding(num_rel, self.dim).to(self.device)
        self.rel_inv_embs = nn.Embedding(num_rel, self.dim).to(self.device)
        sqrt_size = 6.0 / math.sqrt(self.dim)
        nn.init.uniform_(self.ent_h_embs.weight.data, -sqrt_size, sqrt_size)
        nn.init.uniform_(self.ent_t_embs.weight.data, -sqrt_size, sqrt_size)
        nn.init.uniform_(self.rel_embs.weight.data, -sqrt_size, sqrt_size)
        nn.init.uniform_(self.rel_inv_embs.weight.data, -sqrt_size, sqrt_size)

    def _loss(self, pos_scores, neg_scores, neg_ratio):
        P, N = pos_scores.size(0), neg_scores.size(0)
        pos_scores, neg_scores = pos_scores.view(-1).to(self.device), neg_scores.view(-1).to(self.device)
        true_y, corrup_y = torch.ones(P).to(self.device), -torch.ones(N).to(self.device)
        target = torch.cat((true_y, corrup_y), 0)
        y = torch.cat((pos_scores, neg_scores), 0).to(self.device)
        return torch.mean(self.criterion(-target * y))
    
    def forward(self, heads, rels, tails):
        hh_embs = self.ent_h_embs(heads).to(self.device)
        ht_embs = self.ent_h_embs(tails).to(self.device)
        th_embs = self.ent_t_embs(heads).to(self.device)
        tt_embs = self.ent_t_embs(tails).to(self.device)
        r_embs = self.rel_embs(rels).to(self.device)
        r_inv_embs = self.rel_inv_embs(rels).to(self.device)

        scores1 = torch.sum(hh_embs * r_embs * tt_embs, dim=1)
        scores2 = torch.sum(ht_embs * r_inv_embs * th_embs, dim=1)
        return torch.clamp((scores1 + scores2) / 2, -20, 20)

    def _regularization(self, hs, rs, ts): # CUSTOM MADE
        hh_embs = self.ent_h_embs(hs).to(self.device)
        ht_embs = self.ent_h_embs(ts).to(self.device)
        th_embs = self.ent_t_embs(hs).to(self.device)
        tt_embs = self.ent_t_embs(ts).to(self.device)
        r_embs = self.rel_embs(rs).to(self.device)
        r_inv_embs = self.rel_inv_embs(rs).to(self.device)
        regul = (torch.mean(hh_embs ** 2) + torch.mean(ht_embs ** 2) + torch.mean(th_embs ** 2) + torch.mean(tt_embs ** 2) + torch.mean(r_embs ** 2) + torch.mean(r_inv_embs ** 2)) / 6
        return regul

### CONVOLUTIONAL MODELS ###

class ConvKB1D(nn.Module):
    def __init__(self, num_ent, num_rel, args, device, norm=1, hidden_size=100, drop_prob=0.01, kernel_size=1, num_of_filters=64, margin=5.0, lmbda=0.1):
        super(ConvKB1D, self).__init__()
        self.args = args
        self.out_channels = num_of_filters
        self.gamma = margin
        self.num_ent = num_ent
        self.num_rel = num_rel
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.drop_prob = drop_prob
        self.lmbda = lmbda
        self.ent_embeddings = nn.Embedding(self.num_ent, self.hidden_size) 
        self.rel_embeddings = nn.Embedding(self.num_rel, self.hidden_size)
        self.conv1_bn = nn.BatchNorm1d(3)
        self.conv_layer = nn.Conv1d(3, self.out_channels, self.kernel_size)  # kernel size x 3
        self.conv2_bn = nn.BatchNorm1d(self.out_channels)
        self.dropout = nn.Dropout(self.drop_prob)
        self.non_linearity = nn.ReLU()
        self.fc_layer = nn.Linear((self.hidden_size - self.kernel_size + 1) * self.out_channels, 1, bias=False)
        self.criterion = nn.Softplus()
        self.init_parameters()

    def init_parameters(self):
        nn.init.xavier_uniform_(self.ent_embeddings.weight.data)
        nn.init.xavier_uniform_(self.rel_embeddings.weight.data)

        nn.init.xavier_uniform_(self.fc_layer.weight.data)
        nn.init.xavier_uniform_(self.conv_layer.weight.data)

    def _calc(self, e_h, e_r, e_t):
        e_h = e_h.unsqueeze(1) # bs x 1 x dim
        e_r = e_r.unsqueeze(1)
        e_t = e_t.unsqueeze(1)

        conv_input = torch.cat([e_h, e_r, e_t], 1)  # bs x 3 x dim

        conv_input = self.conv1_bn(conv_input)
        out_conv = self.conv_layer(conv_input)
        out_conv = self.conv2_bn(out_conv)
        out_conv = self.non_linearity(out_conv)
        out_conv = out_conv.view(-1, (self.hidden_size - self.kernel_size + 1) * self.out_channels)
        input_fc = self.dropout(out_conv)
        score = self.fc_layer(input_fc).view(-1)

        return -score

    def _loss(self, pos_scores, neg_scores, neg_ratio):
        P, N = pos_scores.size(0), neg_scores.size(0)
        pos_scores, neg_scores = pos_scores.view(-1), neg_scores.view(-1)
        true_y, corrup_y = torch.ones(P), -torch.ones(N)
        target = torch.cat((true_y, corrup_y), 0)
        y = torch.cat((pos_scores, neg_scores), 0)
        tmp = self.criterion((-target * y))
        loss = torch.mean(tmp)
        return loss

    def forward(self, hs, rs, ts):
        e_hs, e_rs, e_ts = self.ent_embeddings(hs), self.rel_embeddings(rs), self.ent_embeddings(ts)
        score = self._calc(e_hs, e_rs, e_ts)
        return score

    def _regularization(self, hs, rs, ts):
        e_hs, e_rs, e_ts = self.ent_embeddings(hs), self.rel_embeddings(rs), self.ent_embeddings(ts)
        l2_reg = torch.mean(e_hs ** 2) + torch.mean(e_ts ** 2) + torch.mean(e_rs ** 2)
        for W in self.conv_layer.parameters():
            l2_reg = l2_reg + W.norm(2)
        for W in self.fc_layer.parameters():
            l2_reg = l2_reg + W.norm(2)
        return l2_reg

class ConvKB2D(nn.Module):
    def __init__(self, num_ent, num_rel, args, device, norm=1, drop_prob=0.5, kernel_size=1, margin=5.0):
        super(ConvKB2D, self).__init__()
        self.args = args
        self.device = device
        self.gamma = margin
        self.num_ent = num_ent
        self.num_rel = num_rel
        self.kernel_size = kernel_size
        self.drop_prob = drop_prob
        self.hidden_size = args.hid_convkb
        self.out_channels = args.num_of_filters_convkb
        self.ent_embeddings = nn.Embedding(self.num_ent, self.hidden_size).to(self.device)
        self.rel_embeddings = nn.Embedding(self.num_rel, self.hidden_size).to(self.device)
        self.conv1_bn = nn.BatchNorm2d(1).to(self.device)
        self.conv_layer = nn.Conv2d(1, self.out_channels, (self.kernel_size, 3)).to(self.device)  # kernel size x 3
        self.conv2_bn = nn.BatchNorm2d(self.out_channels).to(self.device)
        self.dropout = nn.Dropout(self.drop_prob).to(self.device)
        self.non_linearity = nn.ReLU().to(self.device)
        self.fc_layer = nn.Linear((self.hidden_size - self.kernel_size + 1) * self.out_channels, 1, bias=False).to(self.device)
        self.criterion = nn.Softplus()
        self.init_parameters()

    def init_parameters(self):
        nn.init.xavier_uniform_(self.ent_embeddings.weight.data)
        nn.init.xavier_uniform_(self.rel_embeddings.weight.data)
        nn.init.xavier_uniform_(self.fc_layer.weight.data)
        nn.init.xavier_uniform_(self.conv_layer.weight.data)

    def _calc(self, e_h, e_r, e_t):
        e_h = e_h.unsqueeze(1).to(self.device) # bs x 1 x dim
        e_r = e_r.unsqueeze(1).to(self.device)
        e_t = e_t.unsqueeze(1).to(self.device)

        conv_input = torch.cat([e_h, e_r, e_t], 1) # bs x 3 x dim
        conv_input = conv_input.transpose(1, 2)
        # To make tensor of size 4, where second dim is for input channels
        conv_input = conv_input.unsqueeze(1)
        conv_input = self.conv1_bn(conv_input)
        out_conv = self.conv_layer(conv_input)
        out_conv = self.conv2_bn(out_conv)
        out_conv = self.non_linearity(out_conv)
        out_conv = out_conv.view(-1, (self.hidden_size - self.kernel_size + 1) * self.out_channels)
        input_fc = self.dropout(out_conv)
        score = self.fc_layer(input_fc).view(-1)

        return -score

    def _loss(self, pos_scores, neg_scores, neg_ratio):
        P, N = pos_scores.size(0), neg_scores.size(0)
        pos_scores, neg_scores = pos_scores.view(-1).to(self.device), neg_scores.view(-1).to(self.device)
        true_y, corrup_y = torch.ones(P).to(self.device), -torch.ones(N).to(self.device)
        target = torch.cat((true_y, corrup_y), 0)
        y = torch.cat((pos_scores, neg_scores), 0).to(self.device)
        tmp = self.criterion((-target * y))
        loss = torch.mean(tmp)
        return loss

    def forward(self, hs, rs, ts):
        e_hs, e_rs, e_ts = self.ent_embeddings(hs).to(self.device), self.rel_embeddings(rs).to(self.device), self.ent_embeddings(ts).to(self.device)
        score = self._calc(e_hs.to(self.device), e_rs.to(self.device), e_ts.to(self.device))
        return score

    def _regularization(self, hs, rs, ts):
        e_hs, e_rs, e_ts = self.ent_embeddings(hs).to(self.device), self.rel_embeddings(rs).to(self.device), self.ent_embeddings(ts).to(self.device)
        l2_reg = torch.mean(e_hs ** 2) + torch.mean(e_ts ** 2) + torch.mean(e_rs ** 2)
        for W in self.conv_layer.parameters():
            l2_reg = l2_reg + W.norm(2)
        for W in self.fc_layer.parameters():
            l2_reg = l2_reg + W.norm(2)
        return l2_reg

    def loss(self, score, regul):
        return torch.mean(self.criterion(score * self.batch_y)) + self.lmbda * regul

    def predict(self, hs, rs, ts):
        e_hs, e_rs, e_ts = self.ent_embeddings(hs), self.ent_embeddings(rs), self.ent_embeddings(ts)
        score = self._calc(e_hs, e_rs, e_ts)
        return score.data

class ConvE(nn.Module):
    def __init__(self, num_ent, num_rel, args, device):
        super(ConvE, self).__init__()
        self.args = args
        self.device = device
        self.num_ent = num_ent
        self.num_rel = num_rel
        self.ent_embeddings = nn.Embedding(self.num_ent, self.args.dim, padding_idx=0).to(self.device)
        self.rel_embeddings = nn.Embedding(self.num_rel, self.args.dim, padding_idx=0).to(self.device)
        self.inp_drop = nn.Dropout(self.args.input_drop).to(self.device)
        self.hidden_drop = nn.Dropout(self.args.hidden_drop).to(self.device)
        self.feature_map_drop = nn.Dropout2d(self.args.feat_drop).to(self.device)
        self.loss = nn.BCELoss()
        self.emb_dim1 = self.args.embedding_shape1
        self.emb_dim2 = self.args.dim // self.emb_dim1
        nn.init.xavier_uniform_(self.ent_embeddings.weight.data)
        nn.init.xavier_uniform_(self.rel_embeddings.weight.data)
        self.conv1 = nn.Conv2d(1, 32, (3, 3), (1, 1), 0, bias=True).to(self.device)
        self.bn0 = nn.BatchNorm2d(1).to(self.device)
        self.bn1 = nn.BatchNorm2d(32).to(self.device)
        self.bn2 = nn.BatchNorm1d(self.args.dim).to(self.device)
        self.register_parameter('b', nn.Parameter(torch.zeros(self.num_ent).to(self.device)))
        self.fc = nn.Linear(self.args.hidden_size, self.args.dim).to(self.device)

    def calc(self, e1_embedded, rel_embedded):
        stacked_inputs = torch.cat([e1_embedded, rel_embedded], 2) #https://github.com/Cysu/open-reid/issues/69
        stacked_inputs = self.bn0(stacked_inputs)
        x = self.inp_drop(stacked_inputs)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_map_drop(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = torch.mm(x, self.ent_embeddings.weight.transpose(1, 0))
        x += self.b.expand_as(x).to(self.device)
        return x

    def forward(self, hs, rs, ts):
        e1_embedded = self.ent_embeddings(hs).view(-1, 1, self.emb_dim1, self.emb_dim2).to(self.device)
        rel_embedded = self.rel_embeddings(rs).view(-1, 1, self.emb_dim1, self.emb_dim2).to(self.device)
        pred1 = self.calc(e1_embedded, rel_embedded)
        return pred1

    def calc_loss(self, hs, rs, ts):
        targets = self.get_batch(hs.shape[0], ts) 
        e1_embedded = self.ent_embeddings(hs).view(-1, 1, self.emb_dim1, self.emb_dim2).to(self.device)
        rel_embedded = self.rel_embeddings(rs).view(-1, 1, self.emb_dim1, self.emb_dim2).to(self.device)
        pred1 = torch.sigmoid(self.calc(e1_embedded, rel_embedded))
        return self.loss(pred1, targets)

    def _loss(self, y_pos, y_neg, neg_ratio):
        P = y_pos.size(0)
        y_pos = y_pos.view(-1).repeat(neg_ratio).to(self.device)
        y_neg = y_neg.view(-1).to(self.device)
        target = Variable(torch.from_numpy(np.ones(P*neg_ratio, dtype=np.int32))).to(self.device)
        criterion = nn.MarginRankingLoss(self.args.margin)
        loss = criterion(y_pos, y_neg, target)
        return loss

    def get_batch(self, batch_size, batch_t):
        targets = torch.zeros(batch_size, self.num_ent).scatter_(1, batch_t.cpu().view(-1, 1).type(torch.int64), 1).to(self.device)
        return targets

    def get_score(self, h, r, t):
        e1_embedded = self.ent_embeddings(h).view(-1, 1, self.emb_dim1, self.emb_dim2).to(self.device)
        rel_embedded = self.rel_embeddings(r).view(-1, 1, self.emb_dim1, self.emb_dim2).to(self.device)
        pred = self.calc(e1_embedded, rel_embedded)
        return pred
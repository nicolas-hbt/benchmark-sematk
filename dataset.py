import numpy as np
import pandas as pd
import pickle
import random
import torch
import math

class Dataset:
    def __init__(self, ds_name, args):
        self.args = args
        self.name = ds_name
        self.batch_size = args.batch_size
        self.dir = "datasets/" + self.name + "/"
        self.ent2id = self.read_pickle('ent2id')
        self.instype_all = self.read_pickle('ent2classes')
        self.rel2id = self.read_pickle('rel2id')
        self.class2id, self.class2id2ent2id, self.subclassof2id = self.read_pickle('class2id'), self.read_pickle('class2ents'), self.read_pickle('subclassof2id')
        self.r2id2dom2id, self.r2id2range2id = self.read_pickle('rel2dom'), self.read_pickle('rel2range')
        self.r2id2metadom2id, self.r2id2metarange2id = self.read_pickle('rel2metadom'), self.read_pickle('rel2metarange')
        self.setting = args.setting
        self.sem = args.sem
        if self.args.model == 'ConvE' and (self.sem=='schema' or self.sem=='both'):
            inv_r2id2dom2id = {}
            for k,v in self.r2id2dom2id.items():
                try:
                    inv_r2id2dom2id[k + max(self.rel2id.values()) + 1] = self.r2id2range2id[k]
                except:
                    pass
                
            inv_r2id2range2id = {}
            for k,v in self.r2id2range2id.items():
                try:
                    inv_r2id2range2id[k + max(self.rel2id.values()) + 1] = self.r2id2dom2id[k]
                except:
                    pass
            self.r2id2dom2id.update(inv_r2id2dom2id)
            self.r2id2range2id.update(inv_r2id2range2id)

            if 'FB15K' in self.name:
                inv_r2id2metadom2id = {}
                for k,v in self.r2id2metadom2id.items():
                    try:
                        inv_r2id2metadom2id[k + max(self.rel2id.values()) + 1] = self.r2id2metarange2id[k]
                    except:
                        pass
                    
                inv_r2id2metarange2id = {}
                for k,v in self.r2id2metarange2id.items():
                    try:
                        inv_r2id2metarange2id[k + max(self.rel2id.values()) + 1] = self.r2id2metadom2id[k]
                    except:
                        pass
                self.r2id2metadom2id.update(inv_r2id2metadom2id)
                self.r2id2metarange2id.update(inv_r2id2metarange2id)
            
        self.r2hs = self.read_pickle('heads2id')
        if self.args.model == 'ConvE':
            self.r2ts_origin = self.read_pickle('tails2id')
            self.r2ts = self.r2ts_origin.copy()
            for rel, tails in self.r2ts_origin.items():
                self.r2ts[rel+len(self.rel2id)] = self.r2hs[rel]
        else:
            self.r2ts = self.read_pickle('tails2id')
        
        self.data = {}

        if self.args.model == 'ConvE':
            self.data["pd_train"] = pd.read_csv(self.dir + "train2id_inv.txt", sep='\t', header=None, names=['h','r','t'])
            self.data["pd_valid"] = pd.read_csv(self.dir + "valid2id_inv.txt", sep='\t', header=None, names=['h','r','t'])
            self.data["pd_test"] = pd.read_csv(self.dir + "test2id_inv.txt", sep='\t', header=None, names=['h','r','t'])
            self.data["train"] = self.data["pd_train"].to_numpy()
            self.data["valid"] = self.data["pd_valid"].to_numpy()
            self.data["test"] = self.data["pd_test"].to_numpy()
            self.data["df"] = pd.concat([self.data["pd_train"],self.data["pd_valid"],self.data["pd_test"]]).to_numpy()
            self.data["df_lst"] = self.data["df"].tolist()
            self.data["dft"] = tuple(map(tuple, self.data["df"]))
        else:
            self.data["pd_train"] = pd.read_csv(self.dir + "train2id.txt", sep='\t', header=None, names=['h','r','t'])
            self.data["pd_valid"] = pd.read_csv(self.dir + "valid2id.txt", sep='\t', header=None, names=['h','r','t'])
            self.data["pd_test"] = pd.read_csv(self.dir + "test2id.txt", sep='\t', header=None, names=['h','r','t'])
            self.data["train"] = self.data["pd_train"].to_numpy()
            self.data["valid"] = self.data["pd_valid"].to_numpy()
            self.data["test"] = self.data["pd_test"].to_numpy()
            self.data["df"] = pd.concat([self.data["pd_train"],self.data["pd_valid"],self.data["pd_test"]]).to_numpy()
            self.data["df_lst"] = self.data["df"].tolist()
            self.data["dft"] = tuple(map(tuple, self.data["df"]))

        self.neg_ratio = args.neg_ratio
        self.neg_sampler = args.neg_sampler
        self.all_rels = list(self.rel2id.values())
        self.all_ents = list(self.ent2id.values())
        self.batch_index = 0

        self.id2ent = {v:k for k,v in self.ent2id.items()}
        self.id2rel = {v:k for k,v in self.rel2id.items()}
        if self.name not in ['Codex-S', 'Codex-M', 'WN18RR']:
            self.id2class = {v:k for k,v in self.class2id.items()}

        if 'YAGO' in self.name:
            self.dist_classes = self.read_pickle('dist_classes')
            self.dist_classes2id = self.read_pickle('dist_classes2id')

    def inv_rel(self, dataset2id):
        for sub, rel, obj in dataset2id:
            rel_inv = rel + max(self.rel2id.values()) + 1
            dataset2id = np.append(dataset2id, np.array([[obj, rel_inv, sub]]), axis=0)
        return dataset2id

    def read_pickle(self, file):
        try:
            with open(self.dir +  file + ".pkl", 'rb') as f:
                pckl = pickle.load(f)
                return pckl
        except:
            pass
            
    def num_ent(self):
        return len(self.ent2id)
    
    def num_rel(self):
        if self.args.model == 'ConvE':
            return len(self.rel2id)*2
        return len(self.rel2id)

    def num_batch(self):
        return int(math.ceil(float(len(self.data["train"])) / self.batch_size))

    def construct_adj(self):
        edge_index, edge_type = [], []

        for sub, rel, obj in self.data['train']:
            edge_index.append((sub, obj))
            edge_type.append(rel)

        # Adding inverse edges
        for sub, rel, obj in self.data['train']:
            edge_index.append((obj, sub))
            edge_type.append(rel + self.num_rel())

        edge_index  = torch.LongTensor(edge_index).to(self.device).t()
        edge_type   = torch.LongTensor(edge_type). to(self.device)

        return edge_index, edge_type

    def next_pos_batch(self, batch_size):
        if self.batch_index + batch_size < len(self.data["train"]): 
            batch = self.data["train"][self.batch_index: self.batch_index+batch_size]
            self.batch_index += batch_size
        else:
            batch = self.data["train"][self.batch_index:]
            self.batch_index = 0
        return batch
                     
    def random_negative_sampling(self, pos_batch, neg_ratio, side='all'):
        if self.neg_ratio==1:
            neg_ratio=2
        neg_batch = np.repeat(np.copy(pos_batch), neg_ratio, axis=0)
        M = neg_batch.shape[0]
        corr = np.random.randint(self.num_ent() - 1, size=M)
        if side=='all':
            e_idxs = np.random.choice([0, 2], size=M)
            neg_batch[np.arange(M), e_idxs] = corr
        elif side=='tail':
            neg_batch[np.arange(M), 2] = corr
        elif side=='head':
            neg_batch[np.arange(M), 0] = corr
        return self.filtering(pos_batch, neg_batch)

    def tc(self, r2class):
        try:
            return np.random.choice(self.class2id2ent2id[r2class], size=1)
        except KeyError:
            return np.random.choice(self.num_ent() - 1, size=1)

    def anti_tc(self, r2class):
        try:
            other_cl = random.choice(list(self.class2id2ent2id.keys() - [r2class]))
            return random.choice(self.class2id2ent2id[other_cl])
        except KeyError:
            return np.random.choice(self.num_ent() - 1, size=1)

    def tcns(self, pos_batch, neg_ratio, side='all'):
        neg_batch = np.repeat(np.copy(pos_batch), neg_ratio, axis=0)
        M = neg_batch.shape[0]
        if side=='all':
            e_idxs = np.random.choice([0, 2], size=M)
            for i in range(len(neg_batch)):
                h,r,t = neg_batch[i, 0], neg_batch[i, 1], neg_batch[i, 2]
                if e_idxs[i] == 0:
                    neg_batch[i, 0] = self.tc(self.r2id2dom2id[r])
                else:
                    neg_batch[i, 2] = self.tc(self.r2id2range2id[r])
        elif side=='tail':
            for i in range(len(neg_batch)):
                neg_batch[i, 2] = self.tc(self.r2id2range2id[r])
        return self.filtering(pos_batch, neg_batch)

    def meta_tcns(self, pos_batch, neg_ratio, side='all'):
        neg_batch = np.repeat(np.copy(pos_batch), neg_ratio, axis=0)
        M = neg_batch.shape[0]
        if side=='all':
            e_idxs = np.random.choice([0, 2], size=M)
            for i in range(len(neg_batch)):
                h,r,t = neg_batch[i, 0], neg_batch[i, 1], neg_batch[i, 2]
                if e_idxs[i] == 0:
                    neg_batch[i, 0] = self.tc(self.r2id2metadom2id[r])
                else:
                    neg_batch[i, 2] = self.tc(self.r2id2metarange2id[r])
        elif side=='tail':
            for i in range(len(neg_batch)):
                neg_batch[i, 2] = self.tc(self.r2id2metarange2id[r])
        return self.filtering(pos_batch, neg_batch)

    def anti_tcns(self, pos_batch, neg_ratio, side='all'):
        neg_batch = np.repeat(np.copy(pos_batch), neg_ratio, axis=0)
        M = neg_batch.shape[0]
        if side=='all':
            e_idxs = np.random.choice([0, 2], size=M)
            for i in range(len(neg_batch)):
                h,r,t = neg_batch[i, 0], neg_batch[i, 1], neg_batch[i, 2]
                if e_idxs[i] == 0:
                    neg_batch[i, 0] = self.anti_tc(self.r2id2dom2id[r])
                else:
                    neg_batch[i, 2] = self.anti_tc(self.r2id2range2id[r])
        elif side=='tail':
            for i in range(len(neg_batch)):
                neg_batch[i, 2] = self.anti_tc(self.r2id2range2id[r])
        return self.filtering(pos_batch, neg_batch)

    def socher(self, pos_batch, neg_ratio, side='all'):
        neg_batch = np.repeat(np.copy(pos_batch), neg_ratio, axis=0)
        M = neg_batch.shape[0]
        if side=='all':
            e_idxs = np.random.choice([0, 2], size=M)
            for i in range(len(neg_batch)):
                h,r,t = neg_batch[i, 0], neg_batch[i, 1], neg_batch[i, 2]
                if e_idxs[i] == 0:
                    neg_batch[i, 0] = random.choice(list(set(self.H[r]) - set([h])))
                else:
                    neg_batch[i, 2] = random.choice(list(set(self.T[r]) - set([t])))
        elif side=='tail':
            for i in range(len(neg_batch)):
                r,t = neg_batch[i, 1], neg_batch[i, 2]
                neg_batch[i, 2] = random.choice(list(self.set(T[r]) - set([t])))
        return self.filtering(pos_batch, neg_batch)

    def next_batch(self, batch_size, neg_ratio, neg_sampler, device):
        pos_batch = self.next_pos_batch(batch_size)
        if neg_sampler == 'rns':
            neg_batch = self.random_negative_sampling(pos_batch, neg_ratio)
        elif neg_sampler == 'tcns':
            neg_batch = self.type_constrained_sampling(pos_batch, neg_ratio)
        batch = np.append(pos_batch, neg_batch, axis=0)
        batch = torch.tensor(batch)
        return batch

    def was_last_batch(self):
        return (self.batch_index == 0)

    def filtering(self, pos_batch, neg_batch):
        nbt = tuple(map(tuple, neg_batch))
        both = ((set(nbt).intersection((self.data['dft']))))
        dupl = [nbt.index(x) for x in both]
        nbt_rem = tuple(map(tuple, neg_batch[dupl]))
        nbt_ok = tuple(set(nbt) - set(nbt_rem))
        neg_batch = np.asarray(nbt_ok[:(len(nbt_ok)-(len(nbt_ok)%pos_batch.shape[0]))])
        if self.neg_ratio == 1:
            neg_batch = neg_batch[:pos_batch.shape[0]]
        return neg_batch

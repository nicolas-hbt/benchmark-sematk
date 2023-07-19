import torch
from sematch.semantic.graph import DBpediaDataTransform, Taxonomy
from sematch.semantic.similarity import ConceptSimilarity
concept = ConceptSimilarity(Taxonomy(DBpediaDataTransform()), 'models/dbpedia_type_ic.txt')
from functools import reduce
import numpy as np
import time
from tqdm import tqdm
from utils import *
from models import *
from sklearn.utils import shuffle
import pickle
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Tester:
    def __init__(self, dataset, args, model_path, valid_or_test):
        self.args = args
        self.hierarchy = args.hierarchy
        self.name = args.dataset
        self.setting = args.setting
        self.device = args.device
        self.model_name = args.model
        if self.model_name == 'TransE':
            self.model = TransE(dataset.num_ent(), dataset.num_rel(), args.dim, self.device)
        if self.model_name == 'TransH':
            self.model = TransH(dataset.num_ent(), dataset.num_rel(), args.dim, self.device)
        if self.model_name == 'TransR':
            self.model = TransR(dataset.num_ent(), dataset.num_rel(), args.dim, self.device)
        if self.model_name == 'TransD':
            self.model = TransD(dataset.num_ent(), dataset.num_rel(), args.dim, self.device)
        if self.model_name == 'DistMult':
            self.model = DistMult(dataset.num_ent(), dataset.num_rel(), args.dim, self.device)
        if self.model_name == 'ComplEx':
            self.model = ComplEx(dataset.num_ent(), dataset.num_rel(), args.dim, self.device)
        if self.model_name == 'SimplE':
            self.model = SimplE(dataset.num_ent(), dataset.num_rel(), args.dim, self.device, dataset, args)
        if self.model_name == 'SimplE2':
            self.model = SimplE2(dataset.num_ent(), dataset.num_rel(), args.dim, self.device)
        if self.model_name == 'ConvE':
            self.model = ConvE(dataset.num_ent(), dataset.num_rel(), args, self.device)
        if self.model_name == 'ConvKB2D':
            self.model = ConvKB2D(dataset.num_ent(), dataset.num_rel(), args, self.device)
        if self.model_name == 'RGCN':
            self.model = RGCN(dataset.num_ent(), dataset.num_rel(), args, self.device)
        if self.model_name == 'CompGCN':
            self.model = CompGCN_DistMult(dataset.num_ent(), dataset.num_rel(), args, self.device)

        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        self.dataset = dataset
        self.valid_or_test = valid_or_test
        self.batch_size = args.batch_size
        self.neg_ratio = args.neg_ratio
        self.neg_sampler = args.neg_sampler
        self.metric = args.metrics
        self.sem = args.sem
        if self.model_name != 'ConvE':
            with open('datasets/' + self.dataset.name + "/tail2triples.pkl", 'rb') as f:
                self.all_possible_ts = pickle.load(f)
            with open('datasets/' + self.dataset.name + "/head2triples.pkl", 'rb') as f:
                self.all_possible_hs = pickle.load(f)
        else:
            with open('datasets/' + self.dataset.name + "/tail2triples_inv.pkl", 'rb') as f:
                self.all_possible_ts = pickle.load(f)


    def get_observed_h(self, h, r, t):
        return(list(set(self.all_possible_hs[t.item()][r.item()]) - set([h.item()])))

    def get_observed_t(self, h, r, t):
        try:
            return(list(set(self.all_possible_ts[h.item()][r.item()]) - set([t.item()])))
        except KeyError:
            return None

    def predictions(self, h, r, t, all_entities):
        heads = h.reshape(-1, 1).repeat(1, all_entities.size()[1]).to(self.device)
        rels = r.reshape(-1, 1).repeat(1, all_entities.size()[1]).to(self.device)
        tails = t.reshape(-1, 1).repeat(1, all_entities.size()[1]).to(self.device)
        triplets = torch.stack((heads, rels, all_entities), dim=2).reshape(-1, 3).to(self.device)
        tails_predictions = self.model.forward((triplets[:,0]),(triplets[:,1]),(triplets[:,2])).reshape(1, -1)
        triplets = torch.stack((all_entities, rels, tails), dim=2).reshape(-1, 3).to(self.device)
        heads_predictions = self.model.forward((triplets[:,0]),(triplets[:,1]),(triplets[:,2])).reshape(1, -1)
        return heads_predictions.squeeze(), tails_predictions.squeeze()

    def calc_valid_mrr(self):
        schema_OWA = {'sem1_h': 0.0, 'sem3_h': 0.0, 'sem5_h': 0.0, 'sem10_h': 0.0, 'sem1_t': 0.0, 'sem3_t': 0.0, 'sem5_t': 0.0, 'sem10_t': 0.0, \
        'sem1': 0.0, 'sem3': 0.0, 'sem5': 0.0, 'sem10': 0.0}
        sem_t_triples_OWA, sem_h_triples_OWA = 0, 0 # for keeping track of the # of triples for which Sem@K_h/t can be calculated
        schema_CWA = {'sem1_h': 0.0, 'sem3_h': 0.0, 'sem5_h': 0.0, 'sem10_h': 0.0, 'sem1_t': 0.0, 'sem3_t': 0.0, 'sem5_t': 0.0, 'sem10_t': 0.0, \
        'sem1': 0.0, 'sem3': 0.0, 'sem5': 0.0, 'sem10': 0.0}
        schema_CWA_wp = {'sem1_h': 0.0, 'sem3_h': 0.0, 'sem5_h': 0.0, 'sem10_h': 0.0, 'sem1_t': 0.0, 'sem3_t': 0.0, 'sem5_t': 0.0, 'sem10_t': 0.0, \
        'sem1': 0.0, 'sem3': 0.0, 'sem5': 0.0, 'sem10': 0.0}
        sem_t_triples_CWA, sem_h_triples_CWA= 0, 0 # for keeping track of the # of triples for which Sem@K_h/t can be calculated
        ext_CWA = {'sem1_h': 0.0, 'sem3_h': 0.0, 'sem5_h': 0.0, 'sem10_h': 0.0, 'sem1_t': 0.0, 'sem3_t': 0.0, 'sem5_t': 0.0, 'sem10_t': 0.0, \
        'sem1': 0.0, 'sem3': 0.0, 'sem5': 0.0, 'sem10': 0.0}
        sem_t_triples_ext, sem_h_triples_ext= 0, 0 # for keeping track of the # of triples for which Sem@K_h/t can be calculated

        filt_hit1_h, filt_hit1_t, filt_hit3_h, filt_hit3_t, filt_hit5_h, filt_hit5_t, filt_hit10_h, filt_hit10_t = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        cpt_h_na, cpt_t_na, cpt_h_good, cpt_t_good, h1s_h, h1s_t = 0,0,0,0,0,0
        h1_h, h1_t, h1_sem1_h, h1_sem1_t = 0, 0, 0, 0
        filt_mrr_h, filt_mrr_t = [], []
        zero_tensor = torch.tensor([0], device=self.device)
        one_tensor = torch.tensor([1], device=self.device)
        X_valid = torch.from_numpy((self.dataset.data[self.valid_or_test]))
        num_ent = self.dataset.num_ent()
        all_entities = torch.arange(end=num_ent, device=self.device).unsqueeze(0)
        start = time.time()
        if self.model_name =='ConvE':
            half_idx = int(X_valid.shape[0]/2)
            X_valid_tails = X_valid[:half_idx]
            X_valid_tails_inv = X_valid[half_idx:]
            for triple in tqdm(X_valid_tails):
                h,r,t = triple[0], triple[1], triple[2]
                rm_idx_t = self.get_observed_t(h,r,t)
                tails_predictions = self.model.get_score(h.to(self.device),r.to(self.device),t.to(self.device)).squeeze()
                tails_predictions[[rm_idx_t]] = - np.inf
                indices_tail = tails_predictions.argsort(descending=True)
                filt_rank_t = (indices_tail == t).nonzero(as_tuple=True)[0].item() + 1
                filt_mrr_t.append(1.0/filt_rank_t)
                if self.metric == 'sem' or self.metric == 'all':
                    if self.sem == 'schema' or self.sem == 'both':
                        if r.item() in self.dataset.r2id2range2id.keys() and self.dataset.r2id2range2id[r.item()] in self.dataset.class2id2ent2id.keys():
                            if len(self.dataset.class2id2ent2id[self.dataset.r2id2range2id[r.item()]]) >= 10:
                                if self.setting == 'both':
                                    s1, s3, s5, s10 = self.sem_at_k(indices_tail[:100], r.item(), side='tail', k=10, setting='CWA')
                                    schema_CWA['sem1_t'], schema_CWA['sem3_t'], schema_CWA['sem5_t'], schema_CWA['sem10_t'] = schema_CWA['sem1_t']+s1, \
                                    schema_CWA['sem3_t']+s3, schema_CWA['sem5_t']+s5, schema_CWA['sem10_t']+s10
                                    sem_t_triples_CWA+=1
                                    try:
                                        s1, s3, s5, s10 = self.sem_at_k(indices_tail[:100], r.item(), side='tail', k=10, setting='OWA')
                                        schema_OWA['sem1_t'], schema_OWA['sem3_t'], schema_OWA['sem5_t'], schema_OWA['sem10_t'] = schema_OWA['sem1_t']+s1, \
                                        schema_OWA['sem3_t']+s3, schema_OWA['sem5_t']+s5, schema_OWA['sem10_t']+s10
                                        sem_t_triples_OWA+=1
                                    except:
                                        continue
                                elif self.setting=='OWA':
                                    try:
                                        s1, s3, s5, s10 = self.sem_at_k(indices_tail[:100], r.item(), side='tail', k=10, setting='OWA')
                                        schema_OWA['sem1_t'], schema_OWA['sem3_t'], schema_OWA['sem5_t'], schema_OWA['sem10_t'] = schema_OWA['sem1_t']+s1, \
                                        schema_OWA['sem3_t']+s3, schema_OWA['sem5_t']+s5, schema_OWA['sem10_t']+s10
                                        sem_t_triples_OWA+=1
                                    except:
                                        continue
                                elif self.setting=='CWA':
                                        s1, s3, s5, s10 = self.sem_at_k(indices_tail[:100], r.item(), side='tail', k=10, setting='CWA')
                                        schema_CWA['sem1_t'], schema_CWA['sem3_t'], schema_CWA['sem5_t'], schema_CWA['sem10_t'] = schema_CWA['sem1_t']+s1, \
                                        schema_CWA['sem3_t']+s3, schema_CWA['sem5_t']+s5, schema_CWA['sem10_t']+s10
                                        s1, s3, s5, s10 = self.sem_at_k_wp(indices_tail[:100], r.item(), side='tail', k=10, setting='CWA')
                                        schema_CWA_wp['sem1_t'], schema_CWA_wp['sem3_t'], schema_CWA_wp['sem5_t'], schema_CWA_wp['sem10_t'] = schema_CWA_wp['sem1_t']+s1, \
                                        schema_CWA_wp['sem3_t']+s3, schema_CWA_wp['sem5_t']+s5, schema_CWA_wp['sem10_t']+s10
                                        sem_t_triples_CWA+=1

                    if self.sem == 'extensional' or self.sem == 'both': 
                        if r.item() in self.dataset.r2ts.keys():
                            if len(self.dataset.r2ts[r.item()]) >= 10:
                                s1, s3, s5, s10 = self.sem_at_k_ext(indices_tail[:10], r.item(), side='tail', k=10)
                                ext_CWA['sem1_t'], ext_CWA['sem3_t'], ext_CWA['sem5_t'], ext_CWA['sem10_t'] = ext_CWA['sem1_t']+s1, \
                                ext_CWA['sem3_t']+s3, ext_CWA['sem5_t']+s5, ext_CWA['sem10_t']+s10
                                sem_t_triples_ext+=1
                if self.metric == 'ranks' or self.metric == 'all':
                    filt_hit10_t += torch.where(indices_tail[:10] == t.item(), one_tensor, zero_tensor).sum().item()
                    filt_hit5_t += torch.where(indices_tail[:5] == t.item(), one_tensor, zero_tensor).sum().item()
                    filt_hit3_t += torch.where(indices_tail[:3] == t.item(), one_tensor, zero_tensor).sum().item()
                    filt_hit1_t += torch.where(indices_tail[:1] == t.item(), one_tensor, zero_tensor).sum().item()

            # batch of reversed triples        
            for triple in tqdm(X_valid_tails_inv):
                h,r,t = triple[0], triple[1], triple[2]
                rm_idx_t = self.get_observed_t(h,r,t)
                tails_predictions = self.model.get_score(h.to(self.device),r.to(self.device),t.to(self.device)).squeeze()
                tails_predictions[[rm_idx_t]] = - np.inf
                indices_tail = tails_predictions.argsort(descending=True)
                filt_rank_h = (indices_tail == t).nonzero(as_tuple=True)[0].item() + 1
                filt_mrr_h.append(1.0/filt_rank_h)
                if self.metric == 'sem' or self.metric == 'all':
                    if self.sem == 'schema' or self.sem == 'both':
                        if r.item() in self.dataset.r2id2range2id.keys() and self.dataset.r2id2range2id[r.item()] in self.dataset.class2id2ent2id.keys():
                            if len(self.dataset.class2id2ent2id[self.dataset.r2id2range2id[r.item()]]) >= 10:
                                if self.setting == 'both':
                                    s1, s3, s5, s10 = self.sem_at_k(indices_tail[:100], r.item(), side='tail', k=10, setting='CWA')
                                    schema_CWA['sem1_h'], schema_CWA['sem3_h'], schema_CWA['sem5_h'], schema_CWA['sem10_h'] = schema_CWA['sem1_h']+s1, \
                                    schema_CWA['sem3_h']+s3, schema_CWA['sem5_h']+s5, schema_CWA['sem10_h']+s10
                                    sem_h_triples_CWA+=1
                                    try:
                                        s1, s3, s5, s10 = self.sem_at_k(indices_tail[:100], r.item(), side='tail', k=10, setting='OWA')
                                        schema_OWA['sem1_h'], schema_OWA['sem3_h'], schema_OWA['sem5_h'], schema_OWA['sem10_h'] = schema_OWA['sem1_h']+s1, \
                                        schema_OWA['sem3_h']+s3, schema_OWA['sem5_h']+s5, schema_OWA['sem10_h']+s10
                                        sem_h_triples_OWA+=1
                                    except:
                                        continue
                                elif self.setting=='OWA':
                                    try:
                                        s1, s3, s5, s10 = self.sem_at_k(indices_tail[:100], r.item(), side='tail', k=10, setting='OWA')
                                        schema_OWA['sem1_h'], schema_OWA['sem3_h'], schema_OWA['sem5_h'], schema_OWA['sem10_h'] = schema_OWA['sem1_h']+s1, \
                                        schema_OWA['sem3_h']+s3, schema_OWA['sem5_h']+s5, schema_OWA['sem10_h']+s10
                                        sem_h_triples_OWA+=1
                                    except:
                                        continue
                                elif self.setting=='CWA':
                                        s1, s3, s5, s10 = self.sem_at_k(indices_tail[:100], r.item(), side='tail', k=10, setting='CWA')
                                        schema_CWA['sem1_h'], schema_CWA['sem3_h'], schema_CWA['sem5_h'], schema_CWA['sem10_h'] = schema_CWA['sem1_h']+s1, \
                                        schema_CWA['sem3_h']+s3, schema_CWA['sem5_h']+s5, schema_CWA['sem10_h']+s10
                                        s1, s3, s5, s10 = self.sem_at_k_wp(indices_tail[:100], r.item(), side='tail', k=10, setting='CWA')
                                        schema_CWA_wp['sem1_h'], schema_CWA_wp['sem3_h'], schema_CWA_wp['sem5_h'], schema_CWA_wp['sem10_h'] = schema_CWA_wp['sem1_h']+s1, \
                                        schema_CWA_wp['sem3_h']+s3, schema_CWA_wp['sem5_h']+s5, schema_CWA_wp['sem10_h']+s10
                                        sem_h_triples_CWA+=1

                    if self.sem == 'extensional' or self.sem == 'both': 
                        if r.item() in self.dataset.r2ts.keys():
                            if len(self.dataset.r2ts[r.item()]) >= 10:
                                s1, s3, s5, s10 = self.sem_at_k_ext(indices_tail[:10], r.item(), side='tail', k=10)
                                ext_CWA['sem1_h'], ext_CWA['sem3_h'], ext_CWA['sem5_h'], ext_CWA['sem10_h'] = ext_CWA['sem1_h']+s1, \
                                ext_CWA['sem3_h']+s3, ext_CWA['sem5_h']+s5, ext_CWA['sem10_h']+s10
                                sem_h_triples_ext+=1

                if self.metric == 'ranks' or self.metric == 'all':
                    filt_hit10_h += torch.where(indices_tail[:10] == t.item(), one_tensor, zero_tensor).sum().item()
                    filt_hit5_h += torch.where(indices_tail[:5] == t.item(), one_tensor, zero_tensor).sum().item()
                    filt_hit3_h += torch.where(indices_tail[:3] == t.item(), one_tensor, zero_tensor).sum().item()
                    filt_hit1_h += torch.where(indices_tail[:1] == t.item(), one_tensor, zero_tensor).sum().item()

        else:
            X_valid = shuffle(X_valid, random_state=7)
            if X_valid.shape[0] > 10000 :
                X_valid = X_valid[:5000]
            for triple in tqdm(X_valid):
                h,r,t = triple[0], triple[1], triple[2]
                rm_idx_t = self.get_observed_t(h,r,t)
                rm_idx_h = self.get_observed_h(h,r,t)
                heads_predictions, tails_predictions = self.predictions(h,r,t,all_entities)
                # Filtered Scenario
                heads_predictions[[rm_idx_h]], tails_predictions[[rm_idx_t]] = -np.inf, -np.inf
                indices_tail, indices_head = tails_predictions.argsort(descending=True), heads_predictions.argsort(descending=True)
                filt_rank_h = (indices_head == h).nonzero(as_tuple=True)[0].item() + 1
                filt_rank_t = (indices_tail == t).nonzero(as_tuple=True)[0].item() + 1
                if indices_head[0].item() == h.item() :
                   h1_h +=1
                if indices_tail[0].item() == t.item() :
                   h1_t +=1
                # Filtered MR and MRR
                filt_mrr_h.append(1.0/filt_rank_h)
                filt_mrr_t.append(1.0/filt_rank_t)
                if self.metric == 'sem' or self.metric == 'all':
                    if self.sem == 'schema' or self.sem == 'both':
                        if r.item() in self.dataset.r2id2dom2id.keys() and self.dataset.r2id2dom2id[r.item()] in self.dataset.class2id2ent2id.keys():
                            if len(self.dataset.class2id2ent2id[self.dataset.r2id2dom2id[r.item()]]) >= 10:
                                if self.setting == 'both':
                                    s1, s3, s5, s10 = self.sem_at_k(indices_head[:100], r.item(), side='head', k=10, setting='CWA')
                                    schema_CWA['sem1_h'], schema_CWA['sem3_h'], schema_CWA['sem5_h'], schema_CWA['sem10_h'] = schema_CWA['sem1_h']+s1, \
                                    schema_CWA['sem3_h']+s3, schema_CWA['sem5_h']+s5, schema_CWA['sem10_h']+s10
                                    sem_h_triples_CWA+=1

                                    s1, s3, s5, s10 = self.sem_at_k(indices_head[:1000], r.item(), side='head', k=10, setting='OWA')
                                    schema_OWA['sem1_h'], schema_OWA['sem3_h'], schema_OWA['sem5_h'], schema_OWA['sem10_h'] = schema_OWA['sem1_h']+s1, \
                                    schema_OWA['sem3_h']+s3, schema_OWA['sem5_h']+s5, schema_OWA['sem10_h']+s10
                                    sem_h_triples_OWA+=1

                                elif self.setting=='OWA':
                                    try:
                                        s1, s3, s5, s10 = self.sem_at_k(indices_head[:100], r.item(), side='head', k=10, setting='OWA')
                                        schema_OWA['sem1_h'], schema_OWA['sem3_h'], schema_OWA['sem5_h'], schema_OWA['sem10_h'] = schema_OWA['sem1_h']+s1, \
                                        schema_OWA['sem3_h']+s3, schema_OWA['sem5_h']+s5, schema_OWA['sem10_h']+s10
                                        sem_h_triples_OWA+=1
                                    except:
                                        continue
                                elif self.setting=='CWA':
                                    s1, s3, s5, s10 = self.sem_at_k(indices_head[:100], r.item(), side='head', k=10, setting='CWA')
                                    schema_CWA['sem1_h'], schema_CWA['sem3_h'], schema_CWA['sem5_h'], schema_CWA['sem10_h'] = schema_CWA['sem1_h']+s1, \
                                    schema_CWA['sem3_h']+s3, schema_CWA['sem5_h']+s5, schema_CWA['sem10_h']+s10
                                    s1, s3, s5, s10 = self.sem_at_k_wp(indices_head[:100], r.item(), side='head', k=10, setting='CWA')
                                    schema_CWA_wp['sem1_h'], schema_CWA_wp['sem3_h'], schema_CWA_wp['sem5_h'], schema_CWA_wp['sem10_h'] = schema_CWA_wp['sem1_h']+s1, \
                                    schema_CWA_wp['sem3_h']+s3, schema_CWA_wp['sem5_h']+s5, schema_CWA_wp['sem10_h']+s10
                                    sem_h_triples_CWA+=1


                        if r.item() in self.dataset.r2id2range2id.keys() and self.dataset.r2id2range2id[r.item()] in self.dataset.class2id2ent2id.keys():
                            if len(self.dataset.class2id2ent2id[self.dataset.r2id2range2id[r.item()]]) >= 10:
                                if self.setting == 'both':
                                    s1, s3, s5, s10 = self.sem_at_k(indices_tail[:100], r.item(), side='tail', k=10, setting='CWA')
                                    schema_CWA['sem1_t'], schema_CWA['sem3_t'], schema_CWA['sem5_t'], schema_CWA['sem10_t'] = schema_CWA['sem1_t']+s1, \
                                    schema_CWA['sem3_t']+s3, schema_CWA['sem5_t']+s5, schema_CWA['sem10_t']+s10
                                    sem_t_triples_CWA+=1
                                    
                                    s1, s3, s5, s10 = self.sem_at_k(indices_tail[:100], r.item(), side='tail', k=10, setting='OWA')
                                    schema_OWA['sem1_t'], schema_OWA['sem3_t'], schema_OWA['sem5_t'], schema_OWA['sem10_t'] = schema_OWA['sem1_t']+s1, \
                                    schema_OWA['sem3_t']+s3, schema_OWA['sem5_t']+s5, schema_OWA['sem10_t']+s10
                                    sem_t_triples_OWA+=1

                                elif self.setting=='OWA':
                                    try:
                                        s1, s3, s5, s10 = self.sem_at_k(indices_tail[:100], r.item(), side='tail', k=10, setting='OWA')
                                        schema_OWA['sem1_t'], schema_OWA['sem3_t'], schema_OWA['sem5_t'], schema_OWA['sem10_t'] = schema_OWA['sem1_t']+s1, \
                                        schema_OWA['sem3_t']+s3, schema_OWA['sem5_t']+s5, schema_OWA['sem10_t']+s10
                                        sem_t_triples_OWA+=1
                                    except:
                                        continue
                                elif self.setting=='CWA':
                                    s1, s3, s5, s10 = self.sem_at_k(indices_tail[:100], r.item(), side='tail', k=10, setting='CWA')
                                    schema_CWA['sem1_t'], schema_CWA['sem3_t'], schema_CWA['sem5_t'], schema_CWA['sem10_t'] = schema_CWA['sem1_t']+s1, \
                                    schema_CWA['sem3_t']+s3, schema_CWA['sem5_t']+s5, schema_CWA['sem10_t']+s10
                                    s1, s3, s5, s10 = self.sem_at_k_wp(indices_tail[:100], r.item(), side='tail', k=10, setting='CWA')
                                    schema_CWA_wp['sem1_t'], schema_CWA_wp['sem3_t'], schema_CWA_wp['sem5_t'], schema_CWA_wp['sem10_t'] = schema_CWA_wp['sem1_t']+s1, \
                                    schema_CWA_wp['sem3_t']+s3, schema_CWA_wp['sem5_t']+s5, schema_CWA_wp['sem10_t']+s10
                                    sem_t_triples_CWA+=1

                    if self.sem == 'extensional' or self.sem == 'both':
                        if r.item() in self.dataset.r2hs.keys():
                            if len(self.dataset.r2hs[r.item()]) >= 10:
                                s1, s3, s5, s10 = self.sem_at_k_ext(indices_head[:10], r.item(), side='head', k=10)
                                ext_CWA['sem1_h'], ext_CWA['sem3_h'], ext_CWA['sem5_h'], ext_CWA['sem10_h'] = ext_CWA['sem1_h']+s1, \
                                ext_CWA['sem3_h']+s3, ext_CWA['sem5_h']+s5, ext_CWA['sem10_h']+s10
                                sem_h_triples_ext+=1
                        if r.item() in self.dataset.r2ts.keys():
                            if len(self.dataset.r2ts[r.item()]) >= 10:
                                s1, s3, s5, s10 = self.sem_at_k_ext(indices_tail[:10], r.item(), side='tail', k=10)
                                ext_CWA['sem1_t'], ext_CWA['sem3_t'], ext_CWA['sem5_t'], ext_CWA['sem10_t'] = ext_CWA['sem1_t']+s1, \
                                ext_CWA['sem3_t']+s3, ext_CWA['sem5_t']+s5, ext_CWA['sem10_t']+s10
                                sem_t_triples_ext+=1

                if self.metric == 'ranks' or self.metric == 'all':
                    filt_hit10_t += torch.where(indices_tail[:10] == t.item(), one_tensor, zero_tensor).sum().item()
                    filt_hit5_t += torch.where(indices_tail[:5] == t.item(), one_tensor, zero_tensor).sum().item()
                    filt_hit3_t += torch.where(indices_tail[:3] == t.item(), one_tensor, zero_tensor).sum().item()
                    filt_hit1_t += torch.where(indices_tail[:1] == t.item(), one_tensor, zero_tensor).sum().item()
                    filt_hit10_h += torch.where(indices_head[:10] == h.item(), one_tensor, zero_tensor).sum().item()
                    filt_hit5_h += torch.where(indices_head[:5] == h.item(), one_tensor, zero_tensor).sum().item()
                    filt_hit3_h += torch.where(indices_head[:3] == h.item(), one_tensor, zero_tensor).sum().item()
                    filt_hit1_h += torch.where(indices_head[:1] == h.item(), one_tensor, zero_tensor).sum().item()
        print(time.time() - start)
        filt_mrr_t = np.mean(filt_mrr_t)
        filt_mrr_h = np.mean(filt_mrr_h)
        filt_mrr = (filt_mrr_h + filt_mrr_t)/2
        if self.model_name =='ConvE':
            filtered_hits_at_10 = (filt_hit10_h + filt_hit10_t)/(2*X_valid_tails.shape[0])*100
            filtered_hits_at_5 = (filt_hit5_h + filt_hit5_t)/(2*X_valid_tails.shape[0])*100
            filtered_hits_at_3 = (filt_hit3_h + filt_hit3_t)/(2*X_valid_tails.shape[0])*100
            filtered_hits_at_1 = (filt_hit1_h + filt_hit1_t)/(2*X_valid_tails.shape[0])*100
            filt_h10_t, filt_h5_t, filt_h3_t, filt_h1_t = filt_hit10_t/X_valid_tails.shape[0]*100, filt_hit5_t/X_valid_tails.shape[0]*100, filt_hit3_t/X_valid_tails.shape[0]*100, filt_hit1_t/X_valid_tails.shape[0]*100
            filt_h10_h, filt_h5_h, filt_h3_h, filt_h1_h = filt_hit10_h/X_valid_tails.shape[0]*100, filt_hit5_h/X_valid_tails.shape[0]*100, filt_hit3_h/X_valid_tails.shape[0]*100, filt_hit1_h/X_valid_tails.shape[0]*100
        else:
            filtered_hits_at_10 = (filt_hit10_h + filt_hit10_t)/(2*X_valid.shape[0])*100
            filtered_hits_at_5 = (filt_hit5_h + filt_hit5_t)/(2*X_valid.shape[0])*100
            filtered_hits_at_3 = (filt_hit3_h + filt_hit3_t)/(2*X_valid.shape[0])*100
            filtered_hits_at_1 = (filt_hit1_h + filt_hit1_t)/(2*X_valid.shape[0])*100
            filt_h10_t, filt_h5_t, filt_h3_t, filt_h1_t = filt_hit10_t/X_valid.shape[0]*100, filt_hit5_t/X_valid.shape[0]*100, filt_hit3_t/X_valid.shape[0]*100, filt_hit1_t/X_valid.shape[0]*100
            filt_h10_h, filt_h5_h, filt_h3_h, filt_h1_h = filt_hit10_h/X_valid.shape[0]*100, filt_hit5_h/X_valid.shape[0]*100, filt_hit3_h/X_valid.shape[0]*100, filt_hit1_h/X_valid.shape[0]*100

        logger.info('{} MRR: {}'.format('Filtered', filt_mrr))
        logger.info('{} Hits@{}: {}'.format('Filtered', 1, filtered_hits_at_1))
        logger.info('{} Hits@{}: {}'.format('Filtered', 3, filtered_hits_at_3))
        logger.info('{} Hits@{}: {}'.format('Filtered', 10, filtered_hits_at_10))

        if self.sem == 'schema' or self.sem == 'both':
            if self.setting == 'both' or self.setting == 'CWA':
                schema_CWA['sem1'] = (((schema_CWA['sem1_h']/sem_h_triples_CWA) + (schema_CWA['sem1_t']/sem_t_triples_CWA))/2)*100
                schema_CWA['sem3'] = (((schema_CWA['sem3_h']/sem_h_triples_CWA) + (schema_CWA['sem3_t']/sem_t_triples_CWA))/2)*100
                schema_CWA['sem5'] = (((schema_CWA['sem5_h']/sem_h_triples_CWA) + (schema_CWA['sem5_t']/sem_t_triples_CWA))/2)*100
                schema_CWA['sem10'] = (((schema_CWA['sem10_h']/sem_h_triples_CWA) + (schema_CWA['sem10_t']/sem_t_triples_CWA))/2)*100

                schema_CWA['sem1_h'], schema_CWA['sem1_t'] = (schema_CWA['sem1_h']/sem_h_triples_CWA)*100, (schema_CWA['sem1_t']/sem_t_triples_CWA)*100
                schema_CWA['sem3_h'], schema_CWA['sem3_t'] = (schema_CWA['sem3_h']/sem_h_triples_CWA)*100, (schema_CWA['sem3_t']/sem_t_triples_CWA)*100
                schema_CWA['sem5_h'], schema_CWA['sem5_t'] = (schema_CWA['sem5_h']/sem_h_triples_CWA)*100, (schema_CWA['sem5_t']/sem_t_triples_CWA)*100
                schema_CWA['sem10_h'], schema_CWA['sem10_t'] = (schema_CWA['sem10_h']/sem_h_triples_CWA)*100, (schema_CWA['sem10_t']/sem_t_triples_CWA)*100

                schema_CWA_wp['sem1'] = (((schema_CWA_wp['sem1_h']/sem_h_triples_CWA) + (schema_CWA_wp['sem1_t']/sem_t_triples_CWA))/2)*100
                schema_CWA_wp['sem3'] = (((schema_CWA_wp['sem3_h']/sem_h_triples_CWA) + (schema_CWA_wp['sem3_t']/sem_t_triples_CWA))/2)*100
                schema_CWA_wp['sem5'] = (((schema_CWA_wp['sem5_h']/sem_h_triples_CWA) + (schema_CWA_wp['sem5_t']/sem_t_triples_CWA))/2)*100
                schema_CWA_wp['sem10'] = (((schema_CWA_wp['sem10_h']/sem_h_triples_CWA) + (schema_CWA_wp['sem10_t']/sem_t_triples_CWA))/2)*100

                schema_CWA_wp['sem1_h'], schema_CWA_wp['sem1_t'] = (schema_CWA_wp['sem1_h']/sem_h_triples_CWA)*100, (schema_CWA_wp['sem1_t']/sem_t_triples_CWA)*100
                schema_CWA_wp['sem3_h'], schema_CWA_wp['sem3_t'] = (schema_CWA_wp['sem3_h']/sem_h_triples_CWA)*100, (schema_CWA_wp['sem3_t']/sem_t_triples_CWA)*100
                schema_CWA_wp['sem5_h'], schema_CWA_wp['sem5_t'] = (schema_CWA_wp['sem5_h']/sem_h_triples_CWA)*100, (schema_CWA_wp['sem5_t']/sem_t_triples_CWA)*100
                schema_CWA_wp['sem10_h'], schema_CWA_wp['sem10_t'] = (schema_CWA_wp['sem10_h']/sem_h_triples_CWA)*100, (schema_CWA_wp['sem10_t']/sem_t_triples_CWA)*100
                for k in [1, 3, 5, 10]:
                    logger.info('[Schema|CWA] Sem@{}: {}'.format(k, (eval("schema_CWA['sem"+str(k)+"']"))))
                    logger.info('[Schema|CWA|Wu-Palmer] Sem@{}: {}'.format(k, (eval("schema_CWA_wp['sem"+str(k)+"']"))))

            if self.setting == 'both' or self.setting == 'OWA':
                schema_OWA['sem1'] = (((schema_OWA['sem1_h']/sem_h_triples_OWA) + (schema_OWA['sem1_t']/sem_t_triples_OWA))/2)*100
                schema_OWA['sem3'] = (((schema_OWA['sem3_h']/sem_h_triples_OWA) + (schema_OWA['sem3_t']/sem_t_triples_OWA))/2)*100
                schema_OWA['sem5'] = (((schema_OWA['sem5_h']/sem_h_triples_OWA) + (schema_OWA['sem5_t']/sem_t_triples_OWA))/2)*100
                schema_OWA['sem10'] = (((schema_OWA['sem10_h']/sem_h_triples_OWA) + (schema_OWA['sem10_t']/sem_t_triples_OWA))/2)*100

                schema_OWA['sem1_h'], schema_OWA['sem1_t'] = (schema_OWA['sem1_h']/sem_h_triples_OWA)*100, (schema_OWA['sem1_t']/sem_t_triples_OWA)*100
                schema_OWA['sem3_h'], schema_OWA['sem3_t'] = (schema_OWA['sem3_h']/sem_h_triples_OWA)*100, (schema_OWA['sem3_t']/sem_t_triples_OWA)*100
                schema_OWA['sem5_h'], schema_OWA['sem5_t'] = (schema_OWA['sem5_h']/sem_h_triples_OWA)*100, (schema_OWA['sem5_t']/sem_t_triples_OWA)*100
                schema_OWA['sem10_h'], schema_OWA['sem10_t'] = (schema_OWA['sem10_h']/sem_h_triples_OWA)*100, (schema_OWA['sem10_t']/sem_t_triples_OWA)*100
                for k in [1, 3, 5, 10]:
                    logger.info('[Schema|OWA] Sem@{}: {}'.format(k, (eval("schema_OWA['sem"+str(k)+"']"))))

        if self.sem == 'extensional' or self.sem == 'both':
            ext_CWA['sem1'] = (((ext_CWA['sem1_h']/sem_h_triples_ext) + (ext_CWA['sem1_t']/sem_t_triples_ext))/2)*100
            ext_CWA['sem3'] = (((ext_CWA['sem3_h']/sem_h_triples_ext) + (ext_CWA['sem3_t']/sem_t_triples_ext))/2)*100
            ext_CWA['sem5'] = (((ext_CWA['sem5_h']/sem_h_triples_ext) + (ext_CWA['sem5_t']/sem_t_triples_ext))/2)*100
            ext_CWA['sem10'] = (((ext_CWA['sem10_h']/sem_h_triples_ext) + (ext_CWA['sem10_t']/sem_t_triples_ext))/2)*100

            ext_CWA['sem1_h'], ext_CWA['sem1_t'] = (ext_CWA['sem1_h']/sem_h_triples_ext)*100, (ext_CWA['sem1_t']/sem_t_triples_ext)*100
            ext_CWA['sem3_h'], ext_CWA['sem3_t'] = (ext_CWA['sem3_h']/sem_h_triples_ext)*100, (ext_CWA['sem3_t']/sem_t_triples_ext)*100
            ext_CWA['sem5_h'], ext_CWA['sem5_t'] = (ext_CWA['sem5_h']/sem_h_triples_ext)*100, (ext_CWA['sem5_t']/sem_t_triples_ext)*100
            ext_CWA['sem10_h'], ext_CWA['sem10_t'] = (ext_CWA['sem10_h']/sem_h_triples_ext)*100, (ext_CWA['sem10_t']/sem_t_triples_ext)*100
            for k in [1, 3, 5, 10]:
                logger.info('[Extensional] Sem@{}: {}'.format(k, (eval("ext_CWA['sem"+str(k)+"']"))))
                        
        if self.metric == 'sem' or self.metric == 'all':
            return filt_mrr, filtered_hits_at_1, filtered_hits_at_3, filtered_hits_at_5, filtered_hits_at_10, filt_mrr_h, filt_mrr_t, \
            filt_h1_h, filt_h1_t, filt_h3_h, filt_h3_t, filt_h5_h, filt_h5_t, filt_h10_h, filt_h10_t, schema_CWA, schema_OWA, ext_CWA, schema_CWA_wp

        else:
            return filt_mrr, filtered_hits_at_1, filtered_hits_at_3, filtered_hits_at_5, filtered_hits_at_10,\
            filt_mrr_h, filt_mrr_t, filt_h1_h, filt_h1_t, filt_h3_h, filt_h3_t, filt_h5_h, filt_h5_t, filt_h10_h, filt_h10_t

    def test(self):
        schema_OWA = {'sem1_h': 0.0, 'sem3_h': 0.0, 'sem5_h': 0.0, 'sem10_h': 0.0, 'sem1_t': 0.0, 'sem3_t': 0.0, 'sem5_t': 0.0, 'sem10_t': 0.0, \
        'sem1': 0.0, 'sem3': 0.0, 'sem5': 0.0, 'sem10': 0.0}
        sem_t_triples_OWA, sem_h_triples_OWA = 0, 0 # for keeping track of the # of triples for which Sem@K_h/t can be calculated
        schema_CWA = {'sem1_h': 0.0, 'sem3_h': 0.0, 'sem5_h': 0.0, 'sem10_h': 0.0, 'sem1_t': 0.0, 'sem3_t': 0.0, 'sem5_t': 0.0, 'sem10_t': 0.0, \
        'sem1': 0.0, 'sem3': 0.0, 'sem5': 0.0, 'sem10': 0.0}
        schema_CWA_wp = {'sem1_h': 0.0, 'sem3_h': 0.0, 'sem5_h': 0.0, 'sem10_h': 0.0, 'sem1_t': 0.0, 'sem3_t': 0.0, 'sem5_t': 0.0, 'sem10_t': 0.0, \
        'sem1': 0.0, 'sem3': 0.0, 'sem5': 0.0, 'sem10': 0.0}
        sem_t_triples_CWA, sem_h_triples_CWA= 0, 0 # for keeping track of the # of triples for which Sem@K_h/t can be calculated
        ext_CWA = {'sem1_h': 0.0, 'sem3_h': 0.0, 'sem5_h': 0.0, 'sem10_h': 0.0, 'sem1_t': 0.0, 'sem3_t': 0.0, 'sem5_t': 0.0, 'sem10_t': 0.0, \
        'sem1': 0.0, 'sem3': 0.0, 'sem5': 0.0, 'sem10': 0.0}
        sem_t_triples_ext, sem_h_triples_ext= 0, 0 # for keeping track of the # of triples for which Sem@K_h/t can be calculated
        filt_hit1_h, filt_hit1_t, filt_hit3_h, filt_hit3_t, filt_hit5_h, filt_hit5_t, filt_hit10_h, filt_hit10_t = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        filt_mrr_h, filt_mrr_t = [], []
        X_valid_or_test = torch.from_numpy((self.dataset.data[self.valid_or_test]))
        zero_tensor = torch.tensor([0], device=self.device)
        one_tensor = torch.tensor([1], device=self.device)
        num_ent = self.dataset.num_ent()
        all_entities = torch.arange(end=num_ent, device=self.device).unsqueeze(0)
        start = time.time()

        if self.model_name =='ConvE':
            half_idx = int(X_valid_or_test.shape[0]/2)
            X_valid_or_test_tails = X_valid_or_test[:half_idx]
            X_valid_or_test_inv = X_valid_or_test[half_idx:]
            for triple in tqdm(X_valid_or_test_tails):
                h,r,t = triple[0], triple[1], triple[2]
                rm_idx_t = self.get_observed_t(h,r,t)
                tails_predictions = self.model.get_score(h.to(self.device),r.to(self.device),t.to(self.device)).squeeze()
                tails_predictions[[rm_idx_t]] = - np.inf
                indices_tail = tails_predictions.argsort(descending=True)
                filt_rank_t = (indices_tail == t).nonzero(as_tuple=True)[0].item() + 1
                filt_mrr_t.append(1.0/filt_rank_t)
                if self.metric == 'sem' or self.metric == 'all':
                    if self.sem == 'schema' or self.sem == 'both':
                        if r.item() in self.dataset.r2id2range2id.keys() and self.dataset.r2id2range2id[r.item()] in self.dataset.class2id2ent2id.keys():
                            if len(self.dataset.class2id2ent2id[self.dataset.r2id2range2id[r.item()]]) >= 10:
                                if self.setting == 'both':
                                    s1, s3, s5, s10 = self.sem_at_k(indices_tail[:100], r.item(), side='tail', k=10, setting='CWA')
                                    schema_CWA['sem1_t'], schema_CWA['sem3_t'], schema_CWA['sem5_t'], schema_CWA['sem10_t'] = schema_CWA['sem1_t']+s1, \
                                    schema_CWA['sem3_t']+s3, schema_CWA['sem5_t']+s5, schema_CWA['sem10_t']+s10
                                    sem_t_triples_CWA+=1
                                    s1, s3, s5, s10 = self.sem_at_k(indices_tail[:100], r.item(), side='tail', k=10, setting='OWA')
                                    schema_OWA['sem1_t'], schema_OWA['sem3_t'], schema_OWA['sem5_t'], schema_OWA['sem10_t'] = schema_OWA['sem1_t']+s1, \
                                    schema_OWA['sem3_t']+s3, schema_OWA['sem5_t']+s5, schema_OWA['sem10_t']+s10
                                    sem_t_triples_OWA+=1
                                elif self.setting=='OWA':
                                    s1, s3, s5, s10 = self.sem_at_k(indices_tail[:100], r.item(), side='tail', k=10, setting='OWA')
                                    schema_OWA['sem1_t'], schema_OWA['sem3_t'], schema_OWA['sem5_t'], schema_OWA['sem10_t'] = schema_OWA['sem1_t']+s1, \
                                    schema_OWA['sem3_t']+s3, schema_OWA['sem5_t']+s5, schema_OWA['sem10_t']+s10
                                    sem_t_triples_OWA+=1
                                elif self.setting=='CWA':
                                    s1, s3, s5, s10 = self.sem_at_k(indices_tail[:100], r.item(), side='tail', k=10, setting='CWA')
                                    schema_CWA['sem1_t'], schema_CWA['sem3_t'], schema_CWA['sem5_t'], schema_CWA['sem10_t'] = schema_CWA['sem1_t']+s1, \
                                    schema_CWA['sem3_t']+s3, schema_CWA['sem5_t']+s5, schema_CWA['sem10_t']+s10
                                    s1, s3, s5, s10 = self.sem_at_k_wp(indices_tail[:100], r.item(), side='tail', k=10, setting='CWA')
                                    schema_CWA_wp['sem1_t'], schema_CWA_wp['sem3_t'], schema_CWA_wp['sem5_t'], schema_CWA_wp['sem10_t'] = schema_CWA_wp['sem1_t']+s1, \
                                    schema_CWA_wp['sem3_t']+s3, schema_CWA_wp['sem5_t']+s5, schema_CWA_wp['sem10_t']+s10
                                    sem_t_triples_CWA+=1
                    if self.sem == 'extensional' or self.sem == 'both': 
                        if r.item() in self.dataset.r2ts.keys():
                            if len(self.dataset.r2ts[r.item()]) >= 10:
                                s1, s3, s5, s10 = self.sem_at_k_ext(indices_tail[:10], r.item(), side='tail', k=10)
                                ext_CWA['sem1_t'], ext_CWA['sem3_t'], ext_CWA['sem5_t'], ext_CWA['sem10_t'] = ext_CWA['sem1_t']+s1, \
                                ext_CWA['sem3_t']+s3, ext_CWA['sem5_t']+s5, ext_CWA['sem10_t']+s10
                                sem_t_triples_ext+=1
                if self.metric == 'ranks' or self.metric == 'all':
                    filt_hit10_t += torch.where(indices_tail[:10] == t.item(), one_tensor, zero_tensor).sum().item()
                    filt_hit5_t += torch.where(indices_tail[:5] == t.item(), one_tensor, zero_tensor).sum().item()
                    filt_hit3_t += torch.where(indices_tail[:3] == t.item(), one_tensor, zero_tensor).sum().item()
                    filt_hit1_t += torch.where(indices_tail[:1] == t.item(), one_tensor, zero_tensor).sum().item()

            # reversed triples        
            for triple in tqdm(X_valid_or_test_inv):
                h,r,t = triple[0], triple[1], triple[2]
                rm_idx_t = self.get_observed_t(h,r,t)
                tails_predictions = self.model.get_score(h.to(self.device),r.to(self.device),t.to(self.device)).squeeze()
                tails_predictions[[rm_idx_t]] = - np.inf
                indices_tail = tails_predictions.argsort(descending=True)
                filt_rank_h = (indices_tail == t).nonzero(as_tuple=True)[0].item() + 1
                filt_mrr_h.append(1.0/filt_rank_h)
                if self.metric == 'sem' or self.metric == 'all':
                    if self.sem == 'schema' or self.sem == 'both':
                        if r.item() in self.dataset.r2id2range2id.keys() and self.dataset.r2id2range2id[r.item()] in self.dataset.class2id2ent2id.keys():
                            if len(self.dataset.class2id2ent2id[self.dataset.r2id2range2id[r.item()]]) >= 10:
                                if self.setting == 'both':
                                    s1, s3, s5, s10 = self.sem_at_k(indices_tail[:100], r.item(), side='tail', k=10, setting='CWA')
                                    schema_CWA['sem1_h'], schema_CWA['sem3_h'], schema_CWA['sem5_h'], schema_CWA['sem10_h'] = schema_CWA['sem1_h']+s1, \
                                    schema_CWA['sem3_h']+s3, schema_CWA['sem5_h']+s5, schema_CWA['sem10_h']+s10
                                    sem_h_triples_CWA+=1
                                    s1, s3, s5, s10 = self.sem_at_k(indices_tail[:100], r.item(), side='tail', k=10, setting='OWA')
                                    schema_OWA['sem1_h'], schema_OWA['sem3_h'], schema_OWA['sem5_h'], schema_OWA['sem10_h'] = schema_OWA['sem1_h']+s1, \
                                    schema_OWA['sem3_h']+s3, schema_OWA['sem5_h']+s5, schema_OWA['sem10_h']+s10
                                    sem_h_triples_OWA+=1
                                elif self.setting=='OWA':
                                    s1, s3, s5, s10 = self.sem_at_k(indices_tail[:100], r.item(), side='tail', k=10, setting='OWA')
                                    schema_OWA['sem1_h'], schema_OWA['sem3_h'], schema_OWA['sem5_h'], schema_OWA['sem10_h'] = schema_OWA['sem1_h']+s1, \
                                    schema_OWA['sem3_h']+s3, schema_OWA['sem5_h']+s5, schema_OWA['sem10_h']+s10
                                    sem_h_triples_OWA+=1
                                elif self.setting=='CWA':
                                    s1, s3, s5, s10 = self.sem_at_k(indices_tail[:100], r.item(), side='tail', k=10, setting='CWA')
                                    schema_CWA['sem1_h'], schema_CWA['sem3_h'], schema_CWA['sem5_h'], schema_CWA['sem10_h'] = schema_CWA['sem1_h']+s1, \
                                    schema_CWA['sem3_h']+s3, schema_CWA['sem5_h']+s5, schema_CWA['sem10_h']+s10
                                    s1, s3, s5, s10 = self.sem_at_k_wp(indices_tail[:100], r.item(), side='tail', k=10, setting='CWA')
                                    schema_CWA_wp['sem1_h'], schema_CWA_wp['sem3_h'], schema_CWA_wp['sem5_h'], schema_CWA_wp['sem10_h'] = schema_CWA_wp['sem1_h']+s1, \
                                    schema_CWA_wp['sem3_h']+s3, schema_CWA_wp['sem5_h']+s5, schema_CWA_wp['sem10_h']+s10
                                    sem_h_triples_CWA+=1
                    if self.sem == 'extensional' or self.sem == 'both': 
                        if r.item() in self.dataset.r2ts.keys():
                            if len(self.dataset.r2ts[r.item()]) >= 10:
                                s1, s3, s5, s10 = self.sem_at_k_ext(indices_tail[:10], r.item(), side='tail', k=10)
                                ext_CWA['sem1_h'], ext_CWA['sem3_h'], ext_CWA['sem5_h'], ext_CWA['sem10_h'] = ext_CWA['sem1_h']+s1, \
                                ext_CWA['sem3_h']+s3, ext_CWA['sem5_h']+s5, ext_CWA['sem10_h']+s10
                                sem_h_triples_ext+=1

                if self.metric == 'ranks' or self.metric == 'all':
                    filt_hit10_h += torch.where(indices_tail[:10] == t.item(), one_tensor, zero_tensor).sum().item()
                    filt_hit5_h += torch.where(indices_tail[:5] == t.item(), one_tensor, zero_tensor).sum().item()
                    filt_hit3_h += torch.where(indices_tail[:3] == t.item(), one_tensor, zero_tensor).sum().item()
                    filt_hit1_h += torch.where(indices_tail[:1] == t.item(), one_tensor, zero_tensor).sum().item()

        else:
            for triple in tqdm(X_valid_or_test):
                h,r,t = triple[0], triple[1], triple[2]
                rm_idx_t = self.get_observed_t(h,r,t)
                rm_idx_h = self.get_observed_h(h,r,t)
                heads_predictions, tails_predictions = self.predictions(h,r,t,all_entities)
                heads_predictions[[rm_idx_h]], tails_predictions[[rm_idx_t]] = -np.inf, -np.inf #sûr à 100% => OK CORRIGER observed_heads/tails DBpedia
                indices_tail, indices_head = tails_predictions.argsort(descending=True), heads_predictions.argsort(descending=True)
                filt_rank_h = (indices_head == h).nonzero(as_tuple=True)[0].item() + 1
                filt_rank_t = (indices_tail == t).nonzero(as_tuple=True)[0].item() + 1
                # Filtered MR and MRR
                filt_mrr_h.append(1.0/filt_rank_h)
                filt_mrr_t.append(1.0/filt_rank_t)
                if self.metric == 'sem' or self.metric == 'all':
                    if self.sem == 'schema' or self.sem == 'both':
                        if r.item() in self.dataset.r2id2dom2id.keys() and self.dataset.r2id2dom2id[r.item()] in self.dataset.class2id2ent2id.keys():
                            if len(self.dataset.class2id2ent2id[self.dataset.r2id2dom2id[r.item()]]) >= 10:
                                if self.setting == 'both':
                                    s1, s3, s5, s10 = self.sem_at_k(indices_head[:100], r.item(), side='head', k=10, setting='CWA')
                                    schema_CWA['sem1_h'], schema_CWA['sem3_h'], schema_CWA['sem5_h'], schema_CWA['sem10_h'] = schema_CWA['sem1_h']+s1, \
                                    schema_CWA['sem3_h']+s3, schema_CWA['sem5_h']+s5, schema_CWA['sem10_h']+s10
                                    sem_h_triples_CWA+=1
                                    s1, s3, s5, s10 = self.sem_at_k(indices_head[:100], r.item(), side='head', k=10, setting='OWA')
                                    schema_OWA['sem1_h'], schema_OWA['sem3_h'], schema_OWA['sem5_h'], schema_OWA['sem10_h'] = schema_OWA['sem1_h']+s1, \
                                    schema_OWA['sem3_h']+s3, schema_OWA['sem5_h']+s5, schema_OWA['sem10_h']+s10
                                    sem_h_triples_OWA+=1
                                elif self.setting=='OWA':
                                    s1, s3, s5, s10 = self.sem_at_k(indices_head[:100], r.item(), side='head', k=10, setting='OWA')
                                    schema_OWA['sem1_h'], schema_OWA['sem3_h'], schema_OWA['sem5_h'], schema_OWA['sem10_h'] = schema_OWA['sem1_h']+s1, \
                                    schema_OWA['sem3_h']+s3, schema_OWA['sem5_h']+s5, schema_OWA['sem10_h']+s10
                                    sem_h_triples_OWA+=1
                                elif self.setting=='CWA':
                                        s1, s3, s5, s10 = self.sem_at_k(indices_head[:100], r.item(), side='head', k=10, setting='CWA')
                                        schema_CWA['sem1_h'], schema_CWA['sem3_h'], schema_CWA['sem5_h'], schema_CWA['sem10_h'] = schema_CWA['sem1_h']+s1, \
                                        schema_CWA['sem3_h']+s3, schema_CWA['sem5_h']+s5, schema_CWA['sem10_h']+s10
                                        s1, s3, s5, s10 = self.sem_at_k_wp(indices_head[:100], r.item(), side='head', k=10, setting='CWA')
                                        schema_CWA_wp['sem1_h'], schema_CWA_wp['sem3_h'], schema_CWA_wp['sem5_h'], schema_CWA_wp['sem10_h'] = schema_CWA_wp['sem1_h']+s1, \
                                        schema_CWA_wp['sem3_h']+s3, schema_CWA_wp['sem5_h']+s5, schema_CWA_wp['sem10_h']+s10
                                        sem_h_triples_CWA+=1

                        if r.item() in self.dataset.r2id2range2id.keys() and self.dataset.r2id2range2id[r.item()] in self.dataset.class2id2ent2id.keys():
                            if len(self.dataset.class2id2ent2id[self.dataset.r2id2range2id[r.item()]]) >= 10:
                                if self.setting == 'both':
                                    s1, s3, s5, s10 = self.sem_at_k(indices_tail[:100], r.item(), side='tail', k=10, setting='CWA')
                                    schema_CWA['sem1_t'], schema_CWA['sem3_t'], schema_CWA['sem5_t'], schema_CWA['sem10_t'] = schema_CWA['sem1_t']+s1, \
                                    schema_CWA['sem3_t']+s3, schema_CWA['sem5_t']+s5, schema_CWA['sem10_t']+s10
                                    sem_t_triples_CWA+=1
                                    s1, s3, s5, s10 = self.sem_at_k(indices_tail[:100], r.item(), side='tail', k=10, setting='OWA')
                                    schema_OWA['sem1_t'], schema_OWA['sem3_t'], schema_OWA['sem5_t'], schema_OWA['sem10_t'] = schema_OWA['sem1_t']+s1, \
                                    schema_OWA['sem3_t']+s3, schema_OWA['sem5_t']+s5, schema_OWA['sem10_t']+s10
                                    sem_t_triples_OWA+=1
                                elif self.setting=='OWA':
                                    s1, s3, s5, s10 = self.sem_at_k(indices_tail[:100], r.item(), side='tail', k=10, setting='OWA')
                                    schema_OWA['sem1_t'], schema_OWA['sem3_t'], schema_OWA['sem5_t'], schema_OWA['sem10_t'] = schema_OWA['sem1_t']+s1, \
                                    schema_OWA['sem3_t']+s3, schema_OWA['sem5_t']+s5, schema_OWA['sem10_t']+s10
                                    sem_t_triples_OWA+=1
                                elif self.setting=='CWA':
                                    s1, s3, s5, s10 = self.sem_at_k(indices_tail[:100], r.item(), side='tail', k=10, setting='CWA')
                                    schema_CWA['sem1_t'], schema_CWA['sem3_t'], schema_CWA['sem5_t'], schema_CWA['sem10_t'] = schema_CWA['sem1_t']+s1, \
                                    schema_CWA['sem3_t']+s3, schema_CWA['sem5_t']+s5, schema_CWA['sem10_t']+s10
                                    s1, s3, s5, s10 = self.sem_at_k_wp(indices_tail[:100], r.item(), side='tail', k=10, setting='CWA')
                                    schema_CWA_wp['sem1_t'], schema_CWA_wp['sem3_t'], schema_CWA_wp['sem5_t'], schema_CWA_wp['sem10_t'] = schema_CWA_wp['sem1_t']+s1, \
                                    schema_CWA_wp['sem3_t']+s3, schema_CWA_wp['sem5_t']+s5, schema_CWA_wp['sem10_t']+s10
                                    sem_t_triples_CWA+=1

                    if self.sem == 'extensional' or self.sem == 'both':
                        if r.item() in self.dataset.r2hs.keys():
                            if len(self.dataset.r2hs[r.item()]) >= 10:
                                s1, s3, s5, s10 = self.sem_at_k_ext(indices_head[:10], r.item(), side='head', k=10)
                                ext_CWA['sem1_h'], ext_CWA['sem3_h'], ext_CWA['sem5_h'], ext_CWA['sem10_h'] = ext_CWA['sem1_h']+s1, \
                                ext_CWA['sem3_h']+s3, ext_CWA['sem5_h']+s5, ext_CWA['sem10_h']+s10
                                sem_h_triples_ext+=1
                        if r.item() in self.dataset.r2ts.keys():
                            if len(self.dataset.r2ts[r.item()]) >= 10:
                                s1, s3, s5, s10 = self.sem_at_k_ext(indices_tail[:10], r.item(), side='tail', k=10)
                                ext_CWA['sem1_t'], ext_CWA['sem3_t'], ext_CWA['sem5_t'], ext_CWA['sem10_t'] = ext_CWA['sem1_t']+s1, \
                                ext_CWA['sem3_t']+s3, ext_CWA['sem5_t']+s5, ext_CWA['sem10_t']+s10
                                sem_t_triples_ext+=1

                if self.metric == 'ranks' or self.metric == 'all':
                    filt_hit10_t += torch.where(indices_tail[:10] == t.item(), one_tensor, zero_tensor).sum().item()
                    filt_hit5_t += torch.where(indices_tail[:5] == t.item(), one_tensor, zero_tensor).sum().item()
                    filt_hit3_t += torch.where(indices_tail[:3] == t.item(), one_tensor, zero_tensor).sum().item()
                    filt_hit1_t += torch.where(indices_tail[:1] == t.item(), one_tensor, zero_tensor).sum().item()
                    filt_hit10_h += torch.where(indices_head[:10] == h.item(), one_tensor, zero_tensor).sum().item()
                    filt_hit5_h += torch.where(indices_head[:5] == h.item(), one_tensor, zero_tensor).sum().item()
                    filt_hit3_h += torch.where(indices_head[:3] == h.item(), one_tensor, zero_tensor).sum().item()
                    filt_hit1_h += torch.where(indices_head[:1] == h.item(), one_tensor, zero_tensor).sum().item()

        print(time.time() - start)
        filt_mrr_t = np.mean(filt_mrr_t)
        filt_mrr_h = np.mean(filt_mrr_h)
        filt_mrr = (filt_mrr_h + filt_mrr_t)/2
        if self.model_name =='ConvE':
            filtered_hits_at_10 = (filt_hit10_h + filt_hit10_t)/(X_valid_or_test_tails.shape[0]+X_valid_or_test_inv.shape[0])*100
            filtered_hits_at_5 = (filt_hit5_h + filt_hit5_t)/(X_valid_or_test_tails.shape[0]+X_valid_or_test_inv.shape[0])*100
            filtered_hits_at_3 = (filt_hit3_h + filt_hit3_t)/(X_valid_or_test_tails.shape[0]+X_valid_or_test_inv.shape[0])*100
            filtered_hits_at_1 = (filt_hit1_h + filt_hit1_t)/(X_valid_or_test_tails.shape[0]+X_valid_or_test_inv.shape[0])*100
            filt_h10_t, filt_h5_t, filt_h3_t, filt_h1_t = filt_hit10_t/X_valid_or_test_tails.shape[0]*100, filt_hit5_t/X_valid_or_test_tails.shape[0]*100, filt_hit3_t/X_valid_or_test_tails.shape[0]*100, filt_hit1_t/X_valid_or_test_tails.shape[0]*100
            filt_h10_h, filt_h5_h, filt_h3_h, filt_h1_h = filt_hit10_h/X_valid_or_test_inv.shape[0]*100, filt_hit5_h/X_valid_or_test_inv.shape[0]*100, filt_hit3_h/X_valid_or_test_inv.shape[0]*100, filt_hit1_h/X_valid_or_test_inv.shape[0]*100
        else:
            filtered_hits_at_10 = (filt_hit10_h + filt_hit10_t)/(2*X_valid_or_test.shape[0])*100
            filtered_hits_at_5 = (filt_hit5_h + filt_hit5_t)/(2*X_valid_or_test.shape[0])*100
            filtered_hits_at_3 = (filt_hit3_h + filt_hit3_t)/(2*X_valid_or_test.shape[0])*100
            filtered_hits_at_1 = (filt_hit1_h + filt_hit1_t)/(2*X_valid_or_test.shape[0])*100
            filt_h10_t, filt_h5_t, filt_h3_t, filt_h1_t = filt_hit10_t/X_valid_or_test.shape[0]*100, filt_hit5_t/X_valid_or_test.shape[0]*100, filt_hit3_t/X_valid_or_test.shape[0]*100, filt_hit1_t/X_valid_or_test.shape[0]*100
            filt_h10_h, filt_h5_h, filt_h3_h, filt_h1_h = filt_hit10_h/X_valid_or_test.shape[0]*100, filt_hit5_h/X_valid_or_test.shape[0]*100, filt_hit3_h/X_valid_or_test.shape[0]*100, filt_hit1_h/X_valid_or_test.shape[0]*100

        logger.info('{} MRR: {}'.format('Filtered', filt_mrr))
        logger.info('{} Hits@{}: {}'.format('Filtered', 1, filtered_hits_at_1))
        logger.info('{} Hits@{}: {}'.format('Filtered', 3, filtered_hits_at_3))
        logger.info('{} Hits@{}: {}'.format('Filtered', 5, filtered_hits_at_5))
        logger.info('{} Hits@{}: {}'.format('Filtered', 10, filtered_hits_at_10))

        if self.sem == 'schema' or self.sem == 'both':
            if self.setting == 'both' or self.setting == 'CWA':
                print('sem_t_triples_CWA: ', sem_t_triples_CWA)
                print('sem_h_triples_CWA: ', sem_h_triples_CWA)
                schema_CWA['sem1'] = (((schema_CWA['sem1_h']/sem_h_triples_CWA) + (schema_CWA['sem1_t']/sem_t_triples_CWA))/2)*100
                schema_CWA['sem3'] = (((schema_CWA['sem3_h']/sem_h_triples_CWA) + (schema_CWA['sem3_t']/sem_t_triples_CWA))/2)*100
                schema_CWA['sem5'] = (((schema_CWA['sem5_h']/sem_h_triples_CWA) + (schema_CWA['sem5_t']/sem_t_triples_CWA))/2)*100
                schema_CWA['sem10'] = (((schema_CWA['sem10_h']/sem_h_triples_CWA) + (schema_CWA['sem10_t']/sem_t_triples_CWA))/2)*100

                schema_CWA['sem1_h'], schema_CWA['sem1_t'] = (schema_CWA['sem1_h']/sem_h_triples_CWA)*100, (schema_CWA['sem1_t']/sem_t_triples_CWA)*100
                schema_CWA['sem3_h'], schema_CWA['sem3_t'] = (schema_CWA['sem3_h']/sem_h_triples_CWA)*100, (schema_CWA['sem3_t']/sem_t_triples_CWA)*100
                schema_CWA['sem5_h'], schema_CWA['sem5_t'] = (schema_CWA['sem5_h']/sem_h_triples_CWA)*100, (schema_CWA['sem5_t']/sem_t_triples_CWA)*100
                schema_CWA['sem10_h'], schema_CWA['sem10_t'] = (schema_CWA['sem10_h']/sem_h_triples_CWA)*100, (schema_CWA['sem10_t']/sem_t_triples_CWA)*100

                schema_CWA_wp['sem1'] = (((schema_CWA_wp['sem1_h']/sem_h_triples_CWA) + (schema_CWA_wp['sem1_t']/sem_t_triples_CWA))/2)*100
                schema_CWA_wp['sem3'] = (((schema_CWA_wp['sem3_h']/sem_h_triples_CWA) + (schema_CWA_wp['sem3_t']/sem_t_triples_CWA))/2)*100
                schema_CWA_wp['sem5'] = (((schema_CWA_wp['sem5_h']/sem_h_triples_CWA) + (schema_CWA_wp['sem5_t']/sem_t_triples_CWA))/2)*100
                schema_CWA_wp['sem10'] = (((schema_CWA_wp['sem10_h']/sem_h_triples_CWA) + (schema_CWA_wp['sem10_t']/sem_t_triples_CWA))/2)*100

                schema_CWA_wp['sem1_h'], schema_CWA_wp['sem1_t'] = (schema_CWA_wp['sem1_h']/sem_h_triples_CWA)*100, (schema_CWA_wp['sem1_t']/sem_t_triples_CWA)*100
                schema_CWA_wp['sem3_h'], schema_CWA_wp['sem3_t'] = (schema_CWA_wp['sem3_h']/sem_h_triples_CWA)*100, (schema_CWA_wp['sem3_t']/sem_t_triples_CWA)*100
                schema_CWA_wp['sem5_h'], schema_CWA_wp['sem5_t'] = (schema_CWA_wp['sem5_h']/sem_h_triples_CWA)*100, (schema_CWA_wp['sem5_t']/sem_t_triples_CWA)*100
                schema_CWA_wp['sem10_h'], schema_CWA_wp['sem10_t'] = (schema_CWA_wp['sem10_h']/sem_h_triples_CWA)*100, (schema_CWA_wp['sem10_t']/sem_t_triples_CWA)*100
                for k in [1, 3, 5, 10]:
                    logger.info('[Schema|CWA] Sem@{}: {}'.format(k, (eval("schema_CWA['sem"+str(k)+"']"))))
                    logger.info('[Schema|CWA|Wu-Palmer] Sem@{}: {}'.format(k, (eval("schema_CWA_wp['sem"+str(k)+"']"))))

            if self.setting == 'both' or self.setting == 'OWA':
                schema_OWA['sem1'] = (((schema_OWA['sem1_h']/sem_h_triples_OWA) + (schema_OWA['sem1_t']/sem_t_triples_OWA))/2)*100
                schema_OWA['sem3'] = (((schema_OWA['sem3_h']/sem_h_triples_OWA) + (schema_OWA['sem3_t']/sem_t_triples_OWA))/2)*100
                schema_OWA['sem5'] = (((schema_OWA['sem5_h']/sem_h_triples_OWA) + (schema_OWA['sem5_t']/sem_t_triples_OWA))/2)*100
                schema_OWA['sem10'] = (((schema_OWA['sem10_h']/sem_h_triples_OWA) + (schema_OWA['sem10_t']/sem_t_triples_OWA))/2)*100

                schema_OWA['sem1_h'], schema_OWA['sem1_t'] = (schema_OWA['sem1_h']/sem_h_triples_OWA)*100, (schema_OWA['sem1_t']/sem_t_triples_OWA)*100
                schema_OWA['sem3_h'], schema_OWA['sem3_t'] = (schema_OWA['sem3_h']/sem_h_triples_OWA)*100, (schema_OWA['sem3_t']/sem_t_triples_OWA)*100
                schema_OWA['sem5_h'], schema_OWA['sem5_t'] = (schema_OWA['sem5_h']/sem_h_triples_OWA)*100, (schema_OWA['sem5_t']/sem_t_triples_OWA)*100
                schema_OWA['sem10_h'], schema_OWA['sem10_t'] = (schema_OWA['sem10_h']/sem_h_triples_OWA)*100, (schema_OWA['sem10_t']/sem_t_triples_OWA)*100
                for k in [1, 3, 5, 10]:
                    logger.info('[Schema|OWA] Sem@{}: {}'.format(k, (eval("schema_OWA['sem"+str(k)+"']"))))

        if self.sem == 'extensional' or self.sem == 'both':
            ext_CWA['sem1'] = (((ext_CWA['sem1_h']/sem_h_triples_ext) + (ext_CWA['sem1_t']/sem_t_triples_ext))/2)*100
            ext_CWA['sem3'] = (((ext_CWA['sem3_h']/sem_h_triples_ext) + (ext_CWA['sem3_t']/sem_t_triples_ext))/2)*100
            ext_CWA['sem5'] = (((ext_CWA['sem5_h']/sem_h_triples_ext) + (ext_CWA['sem5_t']/sem_t_triples_ext))/2)*100
            ext_CWA['sem10'] = (((ext_CWA['sem10_h']/sem_h_triples_ext) + (ext_CWA['sem10_t']/sem_t_triples_ext))/2)*100
            ext_CWA['sem1_h'], ext_CWA['sem1_t'] = (ext_CWA['sem1_h']/sem_h_triples_ext)*100, (ext_CWA['sem1_t']/sem_t_triples_ext)*100
            ext_CWA['sem3_h'], ext_CWA['sem3_t'] = (ext_CWA['sem3_h']/sem_h_triples_ext)*100, (ext_CWA['sem3_t']/sem_t_triples_ext)*100
            ext_CWA['sem5_h'], ext_CWA['sem5_t'] = (ext_CWA['sem5_h']/sem_h_triples_ext)*100, (ext_CWA['sem5_t']/sem_t_triples_ext)*100
            ext_CWA['sem10_h'], ext_CWA['sem10_t'] = (ext_CWA['sem10_h']/sem_h_triples_ext)*100, (ext_CWA['sem10_t']/sem_t_triples_ext)*100
            for k in [1, 3, 5, 10]:
                logger.info('[Extensional] Sem@{}: {}'.format(k, (eval("ext_CWA['sem"+str(k)+"']"))))
                    
        if self.metric == 'sem' or self.metric == 'all':
            return filt_mrr, filtered_hits_at_1, filtered_hits_at_3, filtered_hits_at_5, filtered_hits_at_10, filt_mrr_h, filt_mrr_t, \
            filt_h1_h, filt_h1_t, filt_h3_h, filt_h3_t, filt_h5_h, filt_h5_t, filt_h10_h, filt_h10_t, schema_CWA, schema_OWA, ext_CWA, schema_CWA_wp

        else:
            return filt_mrr, filtered_hits_at_1, filtered_hits_at_3, filtered_hits_at_5, filtered_hits_at_10,\
            filt_mrr_h, filt_mrr_t, filt_h1_h, filt_h1_t, filt_h3_h, filt_h3_t, filt_h5_h, filt_h5_t, filt_h10_h, filt_h10_t

    def sem_at_k(self, preds, rel, side='head', k=10, setting='CWA'):
        '''
        example: [known, unknown, known, known, unknown, known, known, unknown....]
        in that example, to compute Sem@5, we look at the types of scored entities n° 1, 3, 4, 6 and 7.
        '''
        valid_types = []
        for pred in preds.tolist():
            if len(valid_types) == 10:
                return((valid_types[0]), np.mean(valid_types[:3]), np.mean(valid_types[:5]), np.mean(valid_types[:10]))
            else:
                try:
                    classes = self.dataset.instype_all[pred]
                    if side=='head':
                        dom = self.dataset.r2id2dom2id[rel]
                        valid_types.append(1 if dom in classes else 0)
                    elif side=='tail':
                        rang = self.dataset.r2id2range2id[rel]
                        valid_types.append(1 if rang in classes else 0)
                except KeyError:
                    valid_types.append(0)

    def sem_at_k_wp(self, preds, rel, side='head', k=10, setting='CWA'):
        valid_types = []
        for pred in preds.tolist():
            if len(valid_types) == 10:
                return((valid_types[0]), np.mean(valid_types[:3]), np.mean(valid_types[:5]), np.mean(valid_types[:10]))
            else:
                try:
                    classes = self.dataset.instype_all[pred]
                    if side=='head':
                        dom = self.dataset.r2id2dom2id[rel]
                        if self.name == 'FB15K237':
                            valid_types.append(1 if dom in classes else self.wupalmer_fb15(dom, classes, rel, side))
                        else:
                            valid_types.append(1 if dom in classes else self.wupalmer(dom, classes))
                    elif side=='tail':
                        rang = self.dataset.r2id2range2id[rel]
                        if self.name == 'FB15K237':
                            valid_types.append(1 if rang in classes else self.wupalmer_fb15(rang, classes, rel, side))
                        else:
                            valid_types.append(1 if rang in classes else self.wupalmer(rang, classes))
                except KeyError:
                    continue

    def wupalmer_fb15(self, true_cl, candidate_cls, rel, side):
        if candidate_cls == []:
            return 0.0
        if side=='head':
            return(max([0.5 if candidate_cls[cl] == self.dataset.r2id2metadom2id[rel] else 0 for cl in range(len(candidate_cls))]))
        elif side=='tail':
            return(max([0.5 if candidate_cls[cl] == self.dataset.r2id2metarange2id[rel] else 0 for cl in range(len(candidate_cls))]))

    def wupalmer(self, true_cl, candidate_cls):
        if self.name == 'DB93K':
            truth = (self.dataset.id2class[true_cl])[1:-1]
            wp_lst = [str(concept.similarity(truth, (self.dataset.id2class[cl])[1:-1], 'wup')) for cl in candidate_cls]
            wp_lst = list(map(lambda x: float(x.replace('link error', '0.0')), wp_lst))
            return (max(wp_lst))

        elif 'YAGO' in self.name:
            lca = [self.find_lca(true_cl, cl) for cl in candidate_cls]
            if self.name == 'YAGO3-37K':
                wp_lst = [(2.0*self.dataset.dist_classes2id[lca[i]][37747] / (self.dataset.dist_classes2id[true_cl][lca[i]] + self.dataset.dist_classes2id[candidate_cls[i]][lca[i]] + \
                                                   (2.0*self.dataset.dist_classes2id[lca[i]][37747]))) for i in range(len(candidate_cls))]
            elif self.name == 'YAGO4-18K':
                wp_lst = [(2.0*self.dataset.dist_classes2id[lca[i]][18978] / (self.dataset.dist_classes2id[true_cl][lca[i]] + self.dataset.dist_classes2id[candidate_cls[i]][lca[i]] + \
                                                   (2.0*self.dataset.dist_classes2id[lca[i]][18978]))) for i in range(len(candidate_cls))]
            return (max(wp_lst))

        else:
            print('Wu-Palmer not implemented for this dataset.')
            return 0


    def find_lca(self, c1, c2):
        if c1 in self.dataset.dist_classes2id[c2]:
            lca = c1
        elif c2 in self.dataset.dist_classes2id[c1]:
            lca = c2
        else:
            dict_seq = [self.dataset.dist_classes2id[c1], self.dataset.dist_classes2id[c2]]
            res = reduce(lambda d1,d2: {k: d1.get(k,0)+d2.get(k,0) for k in set(d1)&set(d2)}, dict_seq)
            lca = min(res, key=res.get)
        return lca  
                
        
    def sem_at_k_ext(self, preds, rel, side='head', k=10):
        '''Extensional version of Sem@K'''
        preds = preds[:10]
        if side=='head':
            seen_entities = self.dataset.r2hs[rel]
        else:
            seen_entities = self.dataset.r2ts[rel]
        valid_types = []
        for pred in preds.tolist():
            valid_types.append(1 if pred in (seen_entities) else 0)
            
        return ((valid_types[0]), np.mean(valid_types[:3]), np.mean(valid_types[:5]), np.mean(valid_types[:10]))

    def sem_until(self, preds, rel, side, thresholds = [1,3,5,10]):
        kmax = max(thresholds)
        ks = thresholds.copy()
        for k in thresholds:
            globals()[f'lst{k}'] = []
        if side=='head':
            for pred in preds.tolist():
                if sum(globals()[f'lst{k}']) == kmax:
                    return [len(globals()[f'lst{k}']) for k in thresholds]
                if pred in self.dataset.instype_all:
                    for k in ks:
                        globals()[f'lst{k}'].append(0 if self.dataset.r2id2dom2id[rel] not in (self.dataset.instype_all[pred]) else 1)
                        for k in thresholds:
                            if sum(globals()[f'lst{k}'] == k) : 
                                globals()[f'cpt{k}'] = len(globals()[f'lst{k}'])
                                del ks[k]
                else:
                    continue


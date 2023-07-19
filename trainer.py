from dataset import Dataset
from models import *
from utils import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import os 

class Trainer:
    def __init__(self, dataset, model_name, args):
        self.device = args.device
        self.model_name = model_name
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
        self.dataset = dataset
        self.args = args
        
    def train(self):
        self.model.train()

        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.args.lr
        )

        for epoch in range(1, self.args.ne + 1):
            last_batch = False
            total_loss = 0.0
            while not last_batch:
                if self.model_name == 'ConvE':
                    batch = torch.tensor(self.dataset.next_pos_batch(self.args.batch_size))
                else:
                    batch = self.dataset.next_batch(self.args.batch_size, neg_ratio=self.args.neg_ratio, neg_sampler=self.args.neg_sampler, device = self.device)
                last_batch = self.dataset.was_last_batch()
                optimizer.zero_grad()
                hs  = (batch[:,0]).clone().detach().long().to(self.device)
                rs   = (batch[:,1]).clone().detach().long().to(self.device)
                ts  = (batch[:,2]).clone().detach().long().to(self.device)
                if self.model_name == 'RGCN':
                    train_data = generate_sampled_graph_and_labels(self.dataset.data["train"], self.args.batch_size, self.args.graph_split_size, \
                        self.dataset.num_ent(), self.dataset.num_rel(), self.args.neg_ratio)
                    entity_embedding = self.model(train_data.entity, train_data.edge_index, train_data.edge_type, train_data.edge_norm)
                    loss = self.model.score_loss(entity_embedding, train_data.samples, train_data.labels) + self.args.reg * self.model.reg_loss(entity_embedding)
                elif self.model_name != 'ConvE':
                    scores = self.model.forward(hs, rs, ts)
                    if last_batch:
                        nb_pos = self.dataset.data["train"].shape[0] % self.args.batch_size
                        pos_scores, neg_scores = scores[:nb_pos], scores[nb_pos:]
                    else:
                        pos_scores, neg_scores = scores[:self.args.batch_size], scores[self.args.batch_size:]

                    loss = self.model._loss(pos_scores, neg_scores, self.args.neg_ratio)
                    if self.args.reg != 0.0 :
                        if self.model_name != 'SimplE':
                            loss += self.args.reg*self.model._regularization(batch[:, 0].to(self.device), batch[:, 1].to(self.device), batch[:, 2].to(self.device))
                else:
                    loss = self.model.calc_loss(hs, rs, ts)
                loss.backward()
                optimizer.step()
                total_loss += loss.cpu().item()
        
            if epoch % self.args.save_each == 0:
                print("Loss in iteration " + str(epoch) + ": " + str(total_loss))
                self.save_model(self.model_name, epoch)

    def save_model(self, model, chkpnt):
        print("Saving the model")
        directory = "models/" + self.dataset.name + "/" + model + "/"
        if not os.path.exists(directory):
            os.makedirs(directory)
        torch.save(self.model.state_dict(), directory + "dim="+str(self.args.dim) + \
        "_lr="+str(self.args.lr) + "_neg="+str(self.args.neg_ratio) + "_bs="+str(self.args.batch_size) + "_reg="+str(self.args.reg) + "__epoch="+str(chkpnt) + ".pt")

    def resume_training(self):
        directory = "models/" + self.dataset.name + "/" + self.model_name + "/"
        resume_epoch = self.args.resume_epoch
        if resume_epoch == 0:
            resume_epoch = max([int(f[-11:].split('=')[-1].split('.')[0]) for f in os.listdir("models/" + self.dataset.name + "/" + self.model_name + "/")])
            model_path = directory + "dim="+str(self.args.dim) + \
        "_lr="+str(self.args.lr) + "_neg="+str(self.args.neg_ratio) + "_bs="+str(self.args.batch_size) + "_reg="+str(self.args.reg) + "__epoch="+str(resume_epoch) + ".pt"
        else:
            model_path = directory + "dim="+str(self.args.dim) + \
        "_lr="+str(self.args.lr) + "_neg="+str(self.args.neg_ratio) + "_bs="+str(self.args.batch_size) + "_reg="+str(self.args.reg) + "__epoch="+str(resume_epoch) + ".pt"

        print('Resuming from ' + str(model_path))
        self.model.load_state_dict(torch.load(model_path))
        self.model.train()

        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.args.lr
        )

        for epoch in range(resume_epoch + 1, resume_epoch + self.args.ne + 1):
            if self.model == 'CompGCN':
                loss = run_epoch(epoch, val_mrr = 0)
            else:
                last_batch = False
                total_loss = 0.0
                while not last_batch:
                    if self.model_name == 'ConvE':
                        batch = torch.tensor(self.dataset.next_pos_batch(self.args.batch_size))
                    else:
                        batch = self.dataset.next_batch(self.args.batch_size, neg_ratio=self.args.neg_ratio, neg_sampler=self.args.neg_sampler, device = self.device)
                    last_batch = self.dataset.was_last_batch()
                    optimizer.zero_grad()
                    hs  = (batch[:,0]).clone().detach().long().to(self.device)
                    rs   = (batch[:,1]).clone().detach().long().to(self.device)
                    ts  = (batch[:,2]).clone().detach().long().to(self.device)
                    chunks = self.args.neg_ratio + 1
                    if self.model == 'CompGCN':
                        pred    = self.model.forward(hs, rs)
                        loss    = self.model.loss(pred, label)
                    if self.model_name == 'RGCN':
                        train_data = generate_sampled_graph_and_labels(self.dataset.data["train"], self.args.batch_size, self.args.graph_split_size, \
                            self.dataset.num_ent(), self.dataset.num_rel(), self.args.neg_ratio)
                        entity_embedding = self.model(train_data.entity, train_data.edge_index, train_data.edge_type, train_data.edge_norm)
                        loss = self.model.score_loss(entity_embedding, train_data.samples, train_data.labels) + self.args.reg * self.model.reg_loss(entity_embedding)
                    elif self.model_name != 'ConvE':
                        scores = self.model.forward(hs, rs, ts)
                        if last_batch:
                            nb_pos = self.dataset.data["train"].shape[0] % self.args.batch_size
                            pos_scores, neg_scores = scores[:nb_pos], scores[nb_pos:]
                        else:
                            pos_scores, neg_scores = scores[:self.args.batch_size], scores[self.args.batch_size:]

                        loss = self.model._loss(pos_scores, neg_scores, self.args.neg_ratio)
                        if self.args.reg != 0.0 :
                            loss += self.args.reg*self.model._regularization(batch[:, 0], batch[:, 1], batch[:, 2])
                    else:
                        loss = self.model.calc_loss(hs, rs, ts)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.cpu().item()

                print("Loss in iteration " + str(epoch) + ": " + str(total_loss))
            
                if epoch % self.args.save_each == 0:
                    self.save_model(self.model_name, epoch)
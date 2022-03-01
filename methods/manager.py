from dataloaders.sampler import data_sampler
from dataloaders.data_loader import get_data_loader
from .model import Encoder
from .utils import Moment, dot_dist
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from tqdm import tqdm, trange
from sklearn.cluster import KMeans
from .utils import osdist
class Manager(object):
    def __init__(self, args):
        super().__init__()
        self.id2rel = None
        self.rel2id = None
    def get_proto(self, args, encoder, mem_set):
        # aggregate the prototype set for further use.
        data_loader = get_data_loader(args, mem_set, False, False, 1)

        features = []

        encoder.eval()
        for step, batch_data in enumerate(data_loader):
            labels, tokens, ind = batch_data
            tokens = torch.stack([x.to(args.device) for x in tokens], dim=0)
            with torch.no_grad():
                feature, rep= encoder.bert_forward(tokens)
            features.append(feature)
            self.lbs.append(labels.item())
        features = torch.cat(features, dim=0)

        proto = torch.mean(features, dim=0, keepdim=True)

        return proto, features
    # Use K-Means to select what samples to save, similar to at_least = 0
    def select_data(self, args, encoder, sample_set):
        data_loader = get_data_loader(args, sample_set, shuffle=False, drop_last=False, batch_size=1)
        features = []
        encoder.eval()
        for step, batch_data in enumerate(data_loader):
            labels, tokens, ind = batch_data
            tokens=torch.stack([x.to(args.device) for x in tokens],dim=0)
            with torch.no_grad():
                feature, rp = encoder.bert_forward(tokens)
            features.append(feature.detach().cpu())

        features = np.concatenate(features)
        num_clusters = min(args.num_protos, len(sample_set))
        distances = KMeans(n_clusters=num_clusters, random_state=0).fit_transform(features)

        mem_set = []
        current_feat = []
        for k in range(num_clusters):
            sel_index = np.argmin(distances[:, k])
            instance = sample_set[sel_index]
            mem_set.append(instance)
            current_feat.append(features[sel_index])
        
        current_feat = np.stack(current_feat, axis=0)
        current_feat = torch.from_numpy(current_feat)
        return mem_set, current_feat, current_feat.mean(0)
    
    def get_optimizer(self, args, encoder):
        print('Use {} optim!'.format(args.optim))
        def set_param(module, lr, decay=0):
            parameters_to_optimize = list(module.named_parameters())
            no_decay = ['undecay']
            parameters_to_optimize = [
                {'params': [p for n, p in parameters_to_optimize
                            if not any(nd in n for nd in no_decay)], 'weight_decay': 0.0, 'lr': lr},
                {'params': [p for n, p in parameters_to_optimize
                            if any(nd in n for nd in no_decay)], 'weight_decay': 0.0, 'lr': lr}
            ]
            return parameters_to_optimize
        params = set_param(encoder, args.learning_rate)

        if args.optim == 'adam':
            pytorch_optim = optim.Adam
        else:
            raise NotImplementedError
        optimizer = pytorch_optim(
            params
        )
        return optimizer
    def train_simple_model(self, args, encoder, training_data, epochs):

        data_loader = get_data_loader(args, training_data, shuffle=True)
        encoder.train()

        optimizer = self.get_optimizer(args, encoder)
        def train_data(data_loader_, name = "", is_mem = False):
            losses = []
            td = tqdm(data_loader_, desc=name)
            for step, batch_data in enumerate(td):
                optimizer.zero_grad()
                labels, tokens, ind = batch_data
                labels = labels.to(args.device)
                tokens = torch.stack([x.to(args.device) for x in tokens], dim=0)
                hidden, reps = encoder.bert_forward(tokens)
                loss = self.moment.loss(reps, labels)
                losses.append(loss.item())
                td.set_postfix(loss = np.array(losses).mean())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(encoder.parameters(), args.max_grad_norm)
                optimizer.step()
                # update moemnt
                if is_mem:
                    self.moment.update_mem(ind, reps.detach())
                else:
                    self.moment.update(ind, reps.detach())
            print(f"{name} loss is {np.array(losses).mean()}")
        for epoch_i in range(epochs):
            train_data(data_loader, "init_train_{}".format(epoch_i), is_mem=False)
    def train_mem_model(self, args, encoder, mem_data, proto_mem, epochs, seen_relations):
        history_nums = len(seen_relations) - args.rel_per_task
        if len(proto_mem)>0:
            
            proto_mem = F.normalize(proto_mem, p =2, dim=1)
            dist = dot_dist(proto_mem, proto_mem)
            dist = dist.to(args.device)

        mem_loader = get_data_loader(args, mem_data, shuffle=True)
        encoder.train()
        temp_rel2id = [self.rel2id[x] for x in seen_relations]
        map_relid2tempid = {k:v for v,k in enumerate(temp_rel2id)}
        map_tempid2relid = {k:v for k, v in map_relid2tempid.items()}
        optimizer = self.get_optimizer(args, encoder)
        def train_data(data_loader_, name = "", is_mem = False):
            losses = []
            kl_losses = []
            td = tqdm(data_loader_, desc=name)
            for step, batch_data in enumerate(td):

                optimizer.zero_grad()
                labels, tokens, ind = batch_data
                labels = labels.to(args.device)
                tokens = torch.stack([x.to(args.device) for x in tokens], dim=0)
                zz, reps = encoder.bert_forward(tokens)
                hidden = reps


                need_ratio_compute = ind < history_nums * args.num_protos
                total_need = need_ratio_compute.sum()
                
                if total_need >0 :
                    # Knowledge Distillation for Relieve Forgetting
                    need_ind = ind[need_ratio_compute]
                    need_labels = labels[need_ratio_compute]
                    temp_labels = [map_relid2tempid[x.item()] for x in need_labels]
                    gold_dist = dist[temp_labels]
                    current_proto = self.moment.get_mem_proto()[:history_nums]
                    this_dist = dot_dist(hidden[need_ratio_compute], current_proto.to(args.device))
                    loss1 = self.kl_div_loss(gold_dist, this_dist, t=args.kl_temp)
                    loss1.backward(retain_graph=True)
                else:
                    loss1 = 0.0

                #  Contrastive Replay
                cl_loss = self.moment.loss(reps, labels, is_mem=True, mapping=map_relid2tempid)

                if isinstance(loss1, float):
                    kl_losses.append(loss1)
                else:
                    kl_losses.append(loss1.item())
                loss = cl_loss
                if isinstance(loss, float):
                    losses.append(loss)
                    td.set_postfix(loss = np.array(losses).mean(),  kl_loss = np.array(kl_losses).mean())
                    # update moemnt
                    if is_mem:
                        self.moment.update_mem(ind, reps.detach(), hidden.detach())
                    else:
                        self.moment.update(ind, reps.detach())
                    continue
                losses.append(loss.item())
                td.set_postfix(loss = np.array(losses).mean(),  kl_loss = np.array(kl_losses).mean())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(encoder.parameters(), args.max_grad_norm)
                optimizer.step()
                
                # update moemnt
                if is_mem:
                    self.moment.update_mem(ind, reps.detach())
                else:
                    self.moment.update(ind, reps.detach())
            print(f"{name} loss is {np.array(losses).mean()}")
        for epoch_i in range(epochs):
            train_data(mem_loader, "memory_train_{}".format(epoch_i), is_mem=True)
    def kl_div_loss(self, x1, x2, t=10):

        batch_dist = F.softmax(t * x1, dim=1)
        temp_dist = F.log_softmax(t * x2, dim=1)
        loss = F.kl_div(temp_dist, batch_dist, reduction="batchmean")
        return loss

    @torch.no_grad()
    def evaluate_strict_model(self, args, encoder, test_data, protos4eval, featrues4eval, seen_relations):
        data_loader = get_data_loader(args, test_data, batch_size=1)
        encoder.eval()
        n = len(test_data)
        temp_rel2id = [self.rel2id[x] for x in seen_relations]
        map_relid2tempid = {k:v for v,k in enumerate(temp_rel2id)}
        map_tempid2relid = {k:v for k, v in map_relid2tempid.items()}
        correct = 0
        for step, batch_data in enumerate(data_loader):
            labels, tokens, ind = batch_data
            labels = labels.to(args.device)
            tokens = torch.stack([x.to(args.device) for x in tokens], dim=0)
            hidden, reps = encoder.bert_forward(tokens)
            labels = [map_relid2tempid[x.item()] for x in labels]
            logits = -osdist(hidden, protos4eval)
            seen_relation_ids = [self.rel2id[relation] for relation in seen_relations]
            seen_relation_ids = [map_relid2tempid[x] for x in seen_relation_ids]
            seen_sim = logits[:,seen_relation_ids]
            seen_sim = seen_sim.cpu().data.numpy()
            max_smi = np.max(seen_sim,axis=1)
            label_smi = logits[:,labels].cpu().data.numpy()
            if label_smi >= max_smi:
                correct += 1
        return correct/n

    def train(self, args):
        # set training batch
        for i in range(args.total_round):
            test_cur = []
            test_total = []
            # set random seed
            random.seed(args.seed+i*100)

            # sampler setup
            sampler = data_sampler(args=args, seed=args.seed+i*100)
            self.id2rel = sampler.id2rel
            self.rel2id = sampler.rel2id
            # encoder setup
            encoder = Encoder(args=args).to(args.device)

            # initialize memory and prototypes
            num_class = len(sampler.id2rel)
            memorized_samples = {}

            # load data and start computation
            
            history_relation = []
            proto4repaly = []
            for steps, (training_data, valid_data, test_data, current_relations, historic_test_data, seen_relations) in enumerate(sampler):

                print(current_relations)
                # Initial
                train_data_for_initial = []
                for relation in current_relations:
                    history_relation.append(relation)
                    train_data_for_initial += training_data[relation]
                # train model
                # no memory. first train with current task
                self.moment = Moment(args)
                self.moment.init_moment(args, encoder, train_data_for_initial, is_memory=False)
                self.train_simple_model(args, encoder, train_data_for_initial, args.step1_epochs)

                # repaly
                if len(memorized_samples)>0:
                    # select current task sample
                    for relation in current_relations:
                        memorized_samples[relation], _, _ = self.select_data(args, encoder, training_data[relation])
                    
                    train_data_for_memory = []
                    for relation in history_relation:
                        train_data_for_memory += memorized_samples[relation]
                    
                    self.moment.init_moment(args, encoder, train_data_for_memory, is_memory=True)
                    self.train_mem_model(args, encoder, train_data_for_memory, proto4repaly, args.step2_epochs, seen_relations)

                feat_mem = []
                proto_mem = []

                for relation in current_relations:
                    memorized_samples[relation], feat, temp_proto = self.select_data(args, encoder, training_data[relation])
                    feat_mem.append(feat)
                    proto_mem.append(temp_proto)

                feat_mem = torch.cat(feat_mem, dim=0)
                temp_proto = torch.stack(proto_mem, dim=0)

                protos4eval = []
                featrues4eval = []
                self.lbs = []
                for relation in history_relation:
                    if relation not in current_relations:
                        
                        protos, featrues = self.get_proto(args, encoder, memorized_samples[relation])
                        protos4eval.append(protos)
                        featrues4eval.append(featrues)
                
                if protos4eval:
                    
                    protos4eval = torch.cat(protos4eval, dim=0).detach()
                    protos4eval = torch.cat([protos4eval, temp_proto.to(args.device)], dim=0)

                else:
                    protos4eval = temp_proto.to(args.device)
                proto4repaly = protos4eval.clone()

                test_data_1 = []
                for relation in current_relations:
                    test_data_1 += test_data[relation]

                test_data_2 = []
                for relation in seen_relations:
                    test_data_2 += historic_test_data[relation]

                cur_acc = self.evaluate_strict_model(args, encoder, test_data_1, protos4eval, featrues4eval,seen_relations)
                total_acc = self.evaluate_strict_model(args, encoder, test_data_2, protos4eval, featrues4eval,seen_relations)

                print(f'Restart Num {i+1}')
                print(f'task--{steps + 1}:')
                print(f'current test acc:{cur_acc}')
                print(f'history test acc:{total_acc}')
                test_cur.append(cur_acc)
                test_total.append(total_acc)
                
                print(test_cur)
                print(test_total)
                del self.moment

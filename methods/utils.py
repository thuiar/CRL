from dataloaders.data_loader import get_data_loader
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm, trange
import random

class Moment:
    def __init__(self, args) -> None:
        
        self.labels = None
        self.mem_labels = None
        self.memlen = 0
        self.sample_k = 500
        self.temperature= args.temp
    def get_mem_proto(self):
        c = self._compute_centroids_ind()
        return c
    def _compute_centroids_ind(self):
        cinds = []
        for x in self.mem_labels:
            if x.item() not in cinds:
                cinds.append(x.item())

        num = len(cinds)
        feats = self.mem_features
        centroids = torch.zeros((num, feats.size(1)), dtype=torch.float32, device=feats.device)
        for i, c in enumerate(cinds):
            ind = np.where(self.mem_labels.cpu().numpy() == c)[0]
            centroids[i, :] = F.normalize(feats[ind, :].mean(dim=0), p=2, dim=0)
        return centroids

    def update(self, ind, feature, init=False):
        self.features[ind] = feature
    def update_mem(self, ind, feature, hidden=None):
        self.mem_features[ind] = feature
        if hidden is not None:
            self.hidden_features[ind] = hidden
    @torch.no_grad()
    def init_moment(self, args, encoder, datasets, is_memory=False):
        encoder.eval()
        datalen = len(datasets)
        if not is_memory:
            self.features = torch.zeros(datalen, args.feat_dim).cuda()
            data_loader = get_data_loader(args, datasets)
            td = tqdm(data_loader)
            lbs = []
            for step, batch_data in enumerate(td):

                labels, tokens, ind = batch_data
                tokens = torch.stack([x.to(args.device) for x in tokens], dim=0)
                _, reps = encoder.bert_forward(tokens)
                self.update(ind, reps.detach())
                lbs.append(labels)
            lbs = torch.cat(lbs)
            self.labels = lbs.to(args.device)
        else:
            self.memlen = datalen
            self.mem_features = torch.zeros(datalen, args.feat_dim).cuda()
            self.hidden_features = torch.zeros(datalen, args.encoder_output_size).cuda()
            lbs = []
            data_loader = get_data_loader(args, datasets)
            td = tqdm(data_loader)
            for step, batch_data in enumerate(td):
                labels, tokens, ind = batch_data
                tokens = torch.stack([x.to(args.device) for x in tokens], dim=0)
                hidden, reps = encoder.bert_forward(tokens)
                self.update_mem(ind, reps.detach(), hidden.detach())
                lbs.append(labels)
            lbs = torch.cat(lbs)
            self.mem_labels = lbs.to(args.device)
    
    def loss(self, x, labels, is_mem=False, mapping=None):

        if is_mem:
            ct_x = self.mem_features
            ct_y = self.mem_labels
        else:
            if self.sample_k is not None:
            # sample some instances
                idx = list(range(len(self.features)))
                if len(idx) > self.sample_k:
                    sample_id = random.sample(idx, self.sample_k)
                else:
                    sample_id = idx
                ct_x = self.features[sample_id]
                ct_y = self.labels[sample_id]
            else:
                ct_x = self.features
                ct_y = self.labels

        device = torch.device("cuda") if x.is_cuda else torch.device("cpu")
        dot_product_tempered = torch.mm(x, ct_x.T) / self.temperature  # n * m
        # Minus max for numerical stability with exponential. Same done in cross entropy. Epsilon added to avoid log(0)
        exp_dot_tempered = (
            torch.exp(dot_product_tempered - torch.max(dot_product_tempered, dim=1, keepdim=True)[0].detach()) + 1e-5
        )
        mask_combined = (labels.unsqueeze(1).repeat(1, ct_y.shape[0]) == ct_y).to(device) # n*m
        cardinality_per_samples = torch.sum(mask_combined, dim=1)

        log_prob = -torch.log(exp_dot_tempered / (torch.sum(exp_dot_tempered, dim=1, keepdim=True)))
        supervised_contrastive_loss_per_sample = torch.sum(log_prob * mask_combined, dim=1) / cardinality_per_samples
        supervised_contrastive_loss = torch.mean(supervised_contrastive_loss_per_sample)

        return supervised_contrastive_loss
    
def dot_dist(x1, x2):
    return torch.matmul(x1, x2.t())

def osdist(x, c):
    pairwise_distances_squared = torch.sum(x ** 2, dim=1, keepdim=True) + \
                                 torch.sum(c.t() ** 2, dim=0, keepdim=True) - \
                                 2.0 * torch.matmul(x, c.t())

    error_mask = pairwise_distances_squared <= 0.0

    pairwise_distances = pairwise_distances_squared.clamp(min=1e-16)#.sqrt()

    pairwise_distances = torch.mul(pairwise_distances, ~error_mask)

    return pairwise_distances
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
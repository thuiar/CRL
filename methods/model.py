import torch.nn as nn
import torch
import torch.nn.functional as F
from .backbone import Bert_Encoder

class Encoder(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.encoder = Bert_Encoder(args)
        self.output_size = self.encoder.out_dim
        dim_in = self.output_size
        self.head = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, args.feat_dim)
            )
    def bert_forward(self, x):
        out = self.encoder(x)
        xx = self.head(out)
        xx = F.normalize(xx, p=2, dim=1)
        return out, xx

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchlight import import_class
import numpy as np
from torch.nn.functional import normalize
import math
from torch import Tensor
import random
from itertools import permutations

class SkeletonCLR(nn.Module):
    """ Referring to the code of MOCO, https://arxiv.org/abs/1911.05722 """

    def __init__(self, base_encoder=None, pretrain=True, feature_dim=128, queue_size=32768,
                 momentum=0.999, Temperature=0.07, mlp=True, in_channels=3, hidden_channels=64,
                 hidden_dim=256, num_class=60, dropout=0.5, 
                 graph_args={'layout': 'ntu-rgb+d', 'strategy': 'spatial'},
                 edge_importance_weighting=True, **kwargs):
        """
        K: queue size; number of negative keys (default: 32768)
        m: momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """

        super().__init__()
        base_encoder = import_class(base_encoder)
        self.pretrain = pretrain

        if not self.pretrain:
            self.encoder_q = base_encoder(in_channels=in_channels, hidden_channels=hidden_channels,
                                          hidden_dim=hidden_dim, num_class=num_class,
                                          dropout=dropout, graph_args=graph_args,
                                          edge_importance_weighting=edge_importance_weighting,
                                          no_pretrain=True,
                                          **kwargs)
        else:
            self.K = queue_size
            self.m = momentum
            self.T = Temperature
            self.tem_decay = 0.99999

            self.encoder_q = base_encoder(in_channels=in_channels, hidden_channels=hidden_channels,
                                          hidden_dim=hidden_dim, num_class=feature_dim,
                                          dropout=dropout, graph_args=graph_args,
                                          edge_importance_weighting=edge_importance_weighting,
                                          **kwargs)
            self.encoder_k = base_encoder(in_channels=in_channels, hidden_channels=hidden_channels,
                                          hidden_dim=hidden_dim, num_class=feature_dim,
                                          dropout=dropout, graph_args=graph_args,
                                          edge_importance_weighting=edge_importance_weighting,
                                          **kwargs)


            if mlp:  # hack: brute-force replacement
                dim_mlp = self.encoder_q.fc.weight.shape[1]
                self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp),
                                                  nn.ReLU(),
                                                  self.encoder_q.fc)
                self.encoder_k.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp),
                                                  nn.ReLU(),
                                                  self.encoder_k.fc)

            for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
                param_k.data.copy_(param_q.data)  # initialize
                param_k.requires_grad = False  # not update by gradient

            # create the queue
            self.register_buffer("queue", torch.randn(feature_dim, queue_size))
            self.queue = F.normalize(self.queue, dim=0)
            self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
            
            self.count = torch.zeros(self.K, dtype=torch.int16).cuda()

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        self.count += 1
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        gpu_index = keys.device.index
        self.queue[:, (ptr + batch_size * gpu_index):(ptr + batch_size * (gpu_index + 1))] = keys.T
        self.count[ptr:ptr + batch_size] = 1

    @torch.no_grad()
    def update_ptr(self, batch_size):
        assert self.K % batch_size == 0  # for simplicity
        self.queue_ptr[0] = (self.queue_ptr[0] + batch_size) % self.K

    @torch.no_grad()
    def relative_tempo(self, im_q: Tensor, im_k: Tensor):
        B, C, T, V, M = im_q.shape
        random_indices = torch.randperm(B, device=im_q.device)
        selected_t1 = random_indices[:int(B * 0.5)]
        selected_t2 = random_indices[int(B * 0.5):]

        diff_tempo = random.choice([2])
        T_real = T // diff_tempo
        tempo1 = torch.arange(0, T, 1, device=im_q.device)[: T_real]
        tempo2 = torch.arange(0, T, diff_tempo, device=im_q.device)[ : T_real]
        im_q_real = torch.empty(B, C, T_real, V, M, device=im_q.device)
        im_k_real = torch.empty_like(im_q_real)
        im_k_negative = torch.empty_like(im_q_real)

        im_q_real[selected_t1] = im_q.index_select(0, selected_t1).index_select(2, tempo1)
        im_q_real[selected_t2] = im_q.index_select(0, selected_t2).index_select(2, tempo2)

        im_k_real[selected_t1] = im_k.index_select(0, selected_t1).index_select(2, tempo1)
        im_k_real[selected_t2] = im_k.index_select(0, selected_t2).index_select(2, tempo2)

        im_k_negative[selected_t1] = im_k.index_select(0, selected_t1).index_select(2, tempo2)
        im_k_negative[selected_t2] = im_k.index_select(0, selected_t2).index_select(2, tempo1)

        k_negative_A, k_negative_M = self.encoder_k(im_k_negative)
        k_negative_A = F.normalize(k_negative_A, dim=1)
        k_negative_M = F.normalize(k_negative_M, dim=1)

        return im_q_real, im_k_real, k_negative_A, k_negative_M



    def forward(self, im_q, im_k=None, im_q_dc=None, nnm=False, topk=1):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        """
        if nnm:
            return self.nearest_neighbors_mining(im_q, im_k, im_q_dc, topk)

        if not self.pretrain:
            return self.encoder_q(im_q)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            im_q, im_k, k_neg_A, k_neg_M = self.relative_tempo(im_q, im_k)
            k_A, k_M = self.encoder_k(im_k)  # keys: NxC
            k_A = F.normalize(k_A, dim=1)
            k_M = F.normalize(k_M, dim=1)

        # compute query features
        q_A, q_M = self.encoder_q(im_q)
        q_A = F.normalize(q_A, dim=1)
        q_M = F.normalize(q_M, dim=1)

        q_dc = self.encoder_q(im_q_dc,DC=True)
        q_dc = F.normalize(q_dc, dim=1)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos_A1 = torch.einsum('nc,nc->n', [q_A, k_A]).unsqueeze(-1)
        l_pos_A2 = torch.einsum('nc,nc->n', [q_A, k_neg_A]).unsqueeze(-1)
        # negative logits: NxK
        l_neg_A = torch.einsum('nc,ck->nk', [q_A, self.queue.clone().detach()])

        l_pos_M = torch.einsum('nc,nc->n', [q_M, k_M]).unsqueeze(-1)
        l_neg_M = torch.einsum('nc,nc->n', [q_M, k_neg_M]).unsqueeze(-1)

        l_pos_A1_dc = torch.einsum('nc,nc->n', [q_dc, k_A]).unsqueeze(-1)
        l_neg_A_dc = torch.einsum('nc,ck->nk', [q_dc, self.queue.clone().detach()])

        l_pos_A1 /= self.T
        l_pos_A2 /= self.T
        l_neg_A /= self.T
        l_pos_M /= self.T
        l_neg_M /= self.T
        
        l_pos_A1_dc /= self.T
        l_neg_A_dc /= self.T

        # logits: Nx(1+K)
        logits1 = torch.cat([l_pos_A1, l_neg_A], dim=1)
        logits2 = torch.cat([l_pos_A2, l_neg_A], dim=1)
        logits_M = (l_pos_M, l_neg_M)

        # labels: positive key indicators
        labels_A = torch.zeros(logits1.shape[0], dtype=torch.long).cuda()
        labels_M = torch.ones_like(labels_A)

        logits_A1_dc = torch.cat([l_pos_A1_dc, l_neg_A_dc], dim=1)
        logits_A1_dc = torch.softmax(logits_A1_dc, dim=1)
        labels_dc = logits1.clone().detach()
        labels_dc = torch.softmax(labels_dc, dim=1)
        labels_dc = labels_dc.detach()
        # dequeue and enqueue
        self._dequeue_and_enqueue(k_neg_A)


        return logits1, logits2, labels_A, logits_M, labels_M, logits_A1_dc, labels_dc

    def nearest_neighbors_mining(self, im_q, im_k, im_q_dc, topk=1):

        # Compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            im_q, im_k, k_neg_A, k_neg_M = self.relative_tempo(im_q, im_k)
            k_A, k_M = self.encoder_k(im_k)  # keys: NxC
            k_A = F.normalize(k_A, dim=1)
            k_M = F.normalize(k_M, dim=1)


        # compute query features
        q_A, q_M = self.encoder_q(im_q)
        q_A = F.normalize(q_A, dim=1)
        q_M = F.normalize(q_M, dim=1)

        q_dc = self.encoder_q(im_q_dc,DC=True)
        q_dc = F.normalize(q_dc, dim=1)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos_A1 = torch.einsum('nc,nc->n', [q_A, k_A]).unsqueeze(-1)
        l_pos_A2 = torch.einsum('nc,nc->n', [q_A, k_neg_A]).unsqueeze(-1)
        l_pos_M = torch.einsum('nc,nc->n', [q_M, k_M]).unsqueeze(-1)
        # negative logits: NxK
        l_neg_A = torch.einsum('nc,ck->nk', [q_A, self.queue.clone().detach()])
        l_neg_M = torch.einsum('nc,nc->n', [q_M, k_neg_M]).unsqueeze(-1)

        l_pos_A1_dc = torch.einsum('nc,nc->n', [q_dc, k_A]).unsqueeze(-1)
        l_neg_A_dc = torch.einsum('nc,ck->nk', [q_dc, self.queue.clone().detach()])

        l_pos_A1 /= self.T
        l_pos_A2 /= self.T
        l_neg_A /= self.T
        l_pos_M /= self.T
        l_neg_M /= self.T

        l_pos_A1_dc /= self.T
        l_neg_A_dc /= self.T

        logits1 = torch.cat([l_pos_A1, l_neg_A], dim=1)
        logits2 = torch.cat([l_pos_A2, l_neg_A], dim=1)
        logits_M = (l_pos_M, l_neg_M)

        logits_A1_dc = torch.cat([l_pos_A1_dc, l_neg_A_dc], dim=1)
        logits_A1_dc = torch.softmax(logits_A1_dc, dim=1)

        labels_dc = logits1.clone().detach()
        labels_dc = torch.softmax(labels_dc, dim=1)
        labels_dc = labels_dc.detach()

        # nearest neighbors mining to expand the positive set
        _, topkdix = torch.topk(l_neg_A, topk, dim=1)
        _, topkdix_dc = torch.topk(l_neg_A_dc, topk, dim=1)

        topk_onehot = torch.zeros_like(l_neg_A)
        topk_onehot.scatter_(1, topkdix, 1)
        topk_onehot.scatter_(1, topkdix_dc, 1)

        pos_mask = torch.cat([torch.ones(topk_onehot.size(0), 1).cuda(), topk_onehot], dim=1)
        labels_M = torch.ones_like(torch.zeros(logits1.shape[0], dtype=torch.long).cuda())

        self._dequeue_and_enqueue(k_neg_A)


        return logits1, logits2, pos_mask, logits_M, labels_M, logits_A1_dc, labels_dc

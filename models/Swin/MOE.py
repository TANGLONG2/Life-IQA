import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from typing import Optional
from torch.nn import Parameter
from torch.nn.init import xavier_uniform_, xavier_normal_, constant_

import torch
from torch import nn


class NoisyTopkRouter(nn.Module):
    def __init__(self, n_embed, num_experts, top_k):
        super(NoisyTopkRouter, self).__init__()
        self.top_k = top_k
        self.topkroute_linear = nn.Linear(n_embed, num_experts)
        # add noise
        self.noise_linear = nn.Linear(n_embed, num_experts)

    def forward(self, mh_output):
        # mh_ouput is the output tensor from multihead self attention block
        logits = self.topkroute_linear(mh_output)

        # Noise logits
        noise_logits = self.noise_linear(mh_output)

        # Adding scaled unit gaussian noise to the logits
        noise = torch.randn_like(logits) * F.softplus(noise_logits)
        noisy_logits = logits + noise

        top_k_logits, indices = noisy_logits.topk(self.top_k, dim=-1)
        zeros = torch.full_like(noisy_logits, float('-inf'))
        sparse_logits = zeros.scatter(-1, indices, top_k_logits)
        router_output = F.softmax(sparse_logits, dim=-1)
        return router_output, indices,noisy_logits

class Expert(nn.Module):
    def __init__(self, n_embd,dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(4 * n_embd, n_embd),

        )

    def forward(self, x):
        return self.net(x)

class DeepseekV2MLP(nn.Module):
    def __init__(self, n_embd=384):
        super().__init__()
        self.gate_proj = nn.Linear(n_embd, 4 * n_embd, bias=False)
        self.up_proj = nn.Linear(n_embd, 4 * n_embd, bias=False)
        self.down_proj = nn.Linear(4 * n_embd, n_embd, bias=False)
        
        self.act_fn = nn.SiLU()

    def forward(self, x):
        mlp_out = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return mlp_out



class TopkRouter(nn.Module):
    def __init__(self, n_embed, num_experts, top_k):
        super(TopkRouter, self).__init__()
        self.top_k = top_k
        self.linear = nn.Linear(n_embed, num_experts)

    def forward(self, mh_output):
        logits = self.linear(mh_output)  
        
        top_k_logits, indices = logits.topk(self.top_k, dim=-1)
        
        zeros = torch.full_like(logits, float('-inf'))
        
        sparse_logits = zeros.scatter(-1, indices, top_k_logits)
        
        router_output = F.softmax(sparse_logits, dim=-1)
        return router_output, indices,logits


class SparseMoE(nn.Module):
    def __init__(self, n_embed, num_experts, top_k):
        super(SparseMoE, self).__init__()
        self.n_exp = num_experts
        self.router = TopkRouter(n_embed, num_experts, top_k)
        self.experts = nn.ModuleList([Expert(n_embed) for _ in range(num_experts)])
        self.top_k = top_k

    def forward(self, x):
       
        gating_output,indices,logits = self.router(x)

        aux_loss = self.compute_aux_loss(gating_output,indices)
        z_loss = self.compute_router_z_loss(logits)

        
        final_output = torch.zeros_like(x)

       
        flat_x = x.view(-1, x.size(-1))
        flat_gating_output = gating_output.view(-1, gating_output.size(-1))

        
        for i, expert in enumerate(self.experts):
           
            expert_mask = (indices == i).any(dim=-1)
           
            flat_mask = expert_mask.view(-1)
           
            if flat_mask.any():
                
                expert_input = flat_x[flat_mask]
               
                expert_output = expert(expert_input)

               
                gating_scores = flat_gating_output[flat_mask, i].unsqueeze(1)
               
                weighted_output = expert_output * gating_scores

              
                final_output[expert_mask] += weighted_output.squeeze(1)

        if self.training:
            dummy = 0.0
            for expert in self.experts:
                for p in expert.parameters():
                    dummy = dummy + 0.0 * p.sum()
            final_output = final_output + dummy

        return final_output,aux_loss,z_loss

    def compute_aux_loss(self, expert_probs: torch.Tensor, indices: torch.Tensor):
        """
        Computes Switch Transformer auxiliary loss (https://arxiv.org/abs/2101.03961)
        See equations (4)-(6) on page 7
        """
        """
           expert_probs: (B,T,E) 稀疏softmax后（未选中的专家为0）
           indices:      (B,T,K)  top-k expert id
        """

        # equation (5): compute ratio of tokens allocated to each expert
        # total number of tokens is defined as total tokens in batch * k
        # (k = 1) for the Switch Transformer
        with torch.no_grad():
            one_hot_indices = F.one_hot(indices, num_classes=self.n_exp)  # [B, T, k, n_exp]
            one_hot_indices = torch.sum(one_hot_indices.float(), dim=2)  # [B, T, n_exp] (sum over k dimension)
            tokens_per_expert = torch.mean(one_hot_indices.float(), dim=(0, 1))

        # equation (6): compute ratio of router probability allocated to each expert
        prob_per_expert = torch.mean(expert_probs.float(), dim=(0, 1))

        # equation (4): take a scaled dot product between prob/token allocation vectors
        # multiply the result by the number of experts
        return self.n_exp * torch.sum(prob_per_expert * tokens_per_expert)

    def compute_router_z_loss(self, logits: torch.Tensor):
        """
        Computes ST-MoE router z loss (https://arxiv.org/abs/2202.08906)
        See equation (5) on page 7
        """

        # exponentiate logits, sum logits of each expert, take log, and square
        # code below is the same as:
        # > z_loss = torch.exp(logits)
        # > z_loss = torch.sum(z_loss, dim=-1)
        # > z_loss = torch.log(z_loss) ** 2.0
        z_loss = torch.logsumexp(logits, dim=-1) ** 2.0  # [B, T, n_exp]

        # sum over all tokens and divide by total number of tokens
        return torch.mean(z_loss)





if __name__ == '__main__':
    # net = UPP_Layer(pool_size=7).cuda()
    MOE = SparseMoE(n_embed=384,num_experts=4,top_k=2).cuda()
    # bunch_decoder = TransformerDecoder(bunch_layer, num_layers=1).cuda()
    # input = torch.randn(1, 3, 224, 224).cuda().clone().detach()
    tgt = torch.randn(1, 8, 384).cuda().clone().detach()
    memory = torch.randn(1, 8, 384).cuda().clone().detach()
    output = MOE(tgt)
    print(output.shape)



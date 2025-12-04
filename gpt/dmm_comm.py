import math
import inspect
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from loguru import logger
from torch.nn import functional as F


class LayerNorm(nn.Module): #RMSNorm
    #""" LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """
    # TODO rename class and remove bias
    """ RMSNorm. """

    def __init__(self, ndim, bias=None):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
      #  self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.rms_norm(input, self.weight.shape, self.weight, 1e-5)


class MultiHeadAttention(nn.Module):
    def __init__(self,
                 config,
                 q_dim: int,
                 kv_dim: int = None,
                 n_embd: int = None,
                 ):
        """
        Multi-Head Attention with flexible input/output dimensions.

        Args:
            config : config object with attributes n_embd, n_head, dropout, bias
            q_dim : input dimension for queries
            k_dim : input dimension for keys (defaults to q_dim)
            v_dim : input dimension for values (defaults to k_dim)
        """
        super().__init__()
        self.n_head = config.n_head

        if n_embd is None:
            self.n_embd = config.n_embd
        else:
            self.n_embd = n_embd
        self.dropout = config.dropout

        if kv_dim is None:
            kv_dim = q_dim

        # check divisibility
        assert self.n_embd % self.n_head == 0, f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head})"
        self.head_size = config.n_embd // self.n_head

        # linear projections
        self.q_proj = nn.Linear(q_dim, 2*self.n_embd, bias=config.bias)
        self.kv_proj = nn.Linear(kv_dim, 3*self.n_embd, bias=config.bias)

        # output projection
        self.c_proj = nn.Linear(self.n_embd, q_dim, bias=config.bias)
        self.resid_dropout = nn.Dropout(self.dropout)
        self.attn_dropout = nn.Dropout(self.dropout)

        # per-head normalization
        self.q_norm_1 = LayerNorm(self.head_size)
        self.k_norm_1 = LayerNorm(self.head_size)
        self.q_norm_2 = LayerNorm(self.head_size)
        self.k_norm_2 = LayerNorm(self.head_size)
        self.gr_norm = nn.GroupNorm(self.n_head, self.n_embd)

        self.lmb = nn.Parameter(-4*torch.ones(1, self.n_head, 1, 1))

        # flash attention
        self.flash = hasattr(F, "scaled_dot_product_attention")


    def forward(self, x_q, x_kv=None, attn_bias=None):
        """
        Args:
            x_q : (B, T_q, q_dim) array to turn into queries
            x_kv : (B, T_kv, kv_dim) array to turn into keys and values (defaults to x_q)
            attn_bias : (B, n_head, T_q, T_k) optional attention bias/mask
        Returns:
            y : (B, T_q, n_embd) attended representations
        """
        if x_kv is None:
            x_kv = x_q

        B, T_q, C_q = x_q.size()
        _, T_kv, _ = x_kv.size()

        # linear projections
        q1, q2 = self.q_proj(x_q).chunk(2, dim=2)
        q1 = q1.view(B, T_q, self.n_head, self.head_size).transpose(1, 2)
        q2 = q2.view(B, T_q, self.n_head, self.head_size).transpose(1, 2)

        k1, k2, v = self.kv_proj(x_kv).chunk(3, dim=2)
        k1 = k1.view(B, T_kv, self.n_head, self.head_size).transpose(1, 2)
        k2 = k2.view(B, T_kv, self.n_head, self.head_size).transpose(1, 2)
        v = v.view(B, T_kv, self.n_head, self.head_size).transpose(1, 2)

        # per-head normalization
        q1, q2 = self.q_norm_1(q1), self.q_norm_2(q2)
        k1, k2 = self.k_norm_1(k1), self.k_norm_2(k2)

        if self.flash:
            y1 = F.scaled_dot_product_attention(
                q1, k1, v,
                attn_mask=attn_bias,
                dropout_p=self.dropout if self.training else 0,
                is_causal=False
            )
            y2 = F.scaled_dot_product_attention(
                q2, k2, v,
                attn_mask=attn_bias,
                dropout_p=self.dropout if self.training else 0,
                is_causal=False
            )
            y = y1 - F.softplus(self.lmb)*y2
        else:
            att1 = (q1 @ k1.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_size))
            att2 = (q2 @ k2.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_size))
            if attn_bias is not None:
                att1 = att1 + attn_bias
                att2 = att2 + attn_bias
            att1 = F.softmax(att1, dim=-1)
            att2 = F.softmax(att2, dim=-1)
            att1 = self.attn_dropout(att1)
            att2 = self.attn_dropout(att2)
            y = att1 @ v - F.softplus(self.lmb) * (att2 @ v)

        # reshape and project to output dimension
        y = y.transpose(1, 2).contiguous().view(B, T_q, self.n_embd)
        y = self.gr_norm(y.transpose(1, 2)).transpose(1, 2)
        y = self.resid_dropout(self.c_proj(y))
        return y





class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class SwiGLU_MLP(nn.Module):
    def __init__(self, config, n_embd = None):
        super().__init__()
        if n_embd == None:
            self.n_embd = config.n_embd
        else:
            self.n_embd = n_embd
        self.dropout = config.dropout
        
        hidden_dim = 4 * self.n_embd  # original FFN hidden size
        # Double the hidden dim for fused SwiGLU
        self.c_fc = nn.Linear(self.n_embd, hidden_dim * 2, bias=config.bias)
        self.c_proj = nn.Linear(hidden_dim, self.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(self.dropout)

    def forward(self, x):
        x_fc = self.c_fc(x)                 # shape: [batch, 2*hidden_dim]
        x_a, x_b = x_fc.chunk(2, dim=-1)    # split into two halves
        x = x_a * F.silu(x_b)               # SwiGLU activation
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):

    # TODO
    def __init__(self,
                config,
                q_dim: int, # TODO: ADD TO CONFIG
                kv_dim: int = None,
                n_embd: int = None,
                ):
        super().__init__()
        if n_embd is None:
            self.n_embd = config.n_embd
        else:
            self.n_embd = n_embd

        if kv_dim is None:
            kv_dim = q_dim

      #  self.g_proj = nn.Linear(q_dim, q_dim, bias=config.bias)

        self.ln_1q = LayerNorm(q_dim, bias=config.bias)
        self.ln_1kv = LayerNorm(kv_dim, bias=config.bias)
        self.attn = MultiHeadAttention(
            config, q_dim, kv_dim, n_embd
        )
       # self.attn = MLA_NonCausalSelfAttention(config, q_dim=64)
        self.ln_2 = LayerNorm(q_dim, bias=config.bias)
        self.mlp = SwiGLU_MLP(
            config, q_dim
        )
        #self.mlp = MLP(config)
        # NEW
        self.ln_3 = LayerNorm(q_dim, bias=config.bias)
        self.ln_4 = LayerNorm(q_dim, bias=config.bias)

    def forward(self, x_q, x_kv=None, attn_bias=None):
        if x_kv is None:
            x_kv = x_q
        #x = x + self.attn(self.ln_1(x))
        #x = x + self.mlp(self.ln_2(x))
        #gate = self.g_proj(x_q).sigmoid()
        x = x_q + self.ln_3(self.attn(self.ln_1q(x_q),
                                    self.ln_1kv(x_kv),
                                    attn_bias = attn_bias))
        x = x + self.ln_4(self.mlp(self.ln_2(x)))
        return x


@dataclass
class DMMConfig:
    block_size: int = 256
    vocab_size: int = 67
    field_of_view_size: int = 11*11
    agent_info_size: int = 10
    max_num_neighbors: int = 13

    n_encoder_layer: int = 2 #
    n_decoder_layer: int = 2 #
    n_head: int = 2
    n_embd: int = 16
    latent_embd: int = 8
    latent_tok_n: int = 8
    dropout: float = 0.0
    bias: bool = False
    empty_token_code: int = 66
    action_msg_feats: int = 16
    empty_connection_code: int = -1
    n_comm_rounds: int = 2
    # number of future actions to predict (one head per action)
    num_action_heads: int = 1
    # loss weights for each action head (e.g. [4, 2, 1] for 3 heads)
    # if None, defaults to equal weights (all 1.0)
    loss_weights: Optional[tuple] = None



class RepresentationEncoder(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config
        self.n_head = config.n_head
        self.empty_token_code = config.empty_token_code 
        self.latent_embd = config.latent_embd
        self.latent_tok_n = config.latent_tok_n

        self.max_num_neighbors = config.max_num_neighbors
        self.agent_info_size = config.agent_info_size
        self.field_of_view_size = config.field_of_view_size
        self.block_size = config.block_size

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            wle=nn.Embedding(config.latent_tok_n, config.latent_embd),
            wne=nn.Embedding(config.max_num_neighbors, config.n_embd), # neighbor embeddings
            #NEW several CLS toks as learnable tokens to form encoded representation
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList(
                [Block(config, config.n_embd) for _ in range(config.n_encoder_layer)]
            ),
            latent_encoder=Block(config, 
                                 config.latent_embd,
                                 config.n_embd,
                                 config.n_embd
                                 ),
            ln_f=LayerNorm(config.latent_embd, bias=config.bias),
        ))

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_encoder_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device)  # shape (t)
        latent = torch.arange(0, self.latent_tok_n, dtype=torch.long, device=device)  # shape (t)
        
        # mask empty tokens out
        empty_mask_idx = (idx == self.empty_token_code)
        # B , T --> B, T, T
        attn_bias = torch.zeros(b, self.n_head, t, t, device=device)
        attn_bias = attn_bias.masked_fill(empty_mask_idx[:, None, None, :], float('-inf'))

        #create neighbor embeddings (includes agent itself as a neighbor)
        if self.agent_info_size == 0:
            # Directions observation: no explicit neighbor-info segment.
            # Just create a zero bias of the same length as the sequence.
            nbrs_embd = torch.zeros(t, self.transformer.wte.weight.size(1),
                                    device=device, dtype=self.transformer.wte.weight.dtype)
        else:
            nbrs = torch.arange(self.max_num_neighbors, device=device, dtype=torch.long)
            nbrs = self.transformer.wne(nbrs.repeat_interleave(self.agent_info_size))
            # Use actual sequence length t so nbrs_embd matches tok_emb/pos_emb along time
            tail_size = t - self.field_of_view_size - self.max_num_neighbors * self.agent_info_size
            assert tail_size >= 0, (
                f"Sequence length {t} is too small. "
                f"Need at least {self.field_of_view_size + self.max_num_neighbors * self.agent_info_size} tokens, "
                f"but got {t}"
            )
            nbrs_embd = torch.cat(
                [
                    torch.zeros(self.field_of_view_size, nbrs.shape[-1], device=device, dtype=nbrs.dtype),
                    nbrs,
                    torch.zeros(tail_size, nbrs.shape[-1], device=device, dtype=nbrs.dtype),
                ],
                dim=0,
            )

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos)  # position embeddings of shape (t, n_embd)
        latent_emb = self.transformer.wle(latent) # position embeddings of shape (latent_tok_n, latent_tok_embd)
        latent_emb = latent_emb.unsqueeze(0).repeat(b,1,1)

        x = self.transformer.drop(tok_emb + pos_emb + nbrs_embd)
        latent = self.transformer.drop(latent_emb)

        for block in self.transformer.h:
            x = block(x, attn_bias = attn_bias)

        # empty mask for making latent
        empty_mask_latent = empty_mask_idx[:, None, :].repeat(1, self.latent_tok_n, 1) # B , T --> B, T_latent, T
        attn_bias = torch.zeros_like(empty_mask_latent, dtype=torch.float, device=device)
        attn_bias = attn_bias.masked_fill(empty_mask_latent, float('-inf'))
        attn_bias = attn_bias[:, None, :, :].repeat(1, self.n_head, 1, 1)

        latent = self.transformer.latent_encoder(latent, x, attn_bias = attn_bias)
        latent = self.transformer.ln_f(latent)
        return latent


# TODO: add agent identity info
class MessageCoordinator(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config
        assert config.empty_connection_code == -1
        self.empty_msg_emb = nn.Embedding(1, config.latent_embd)
        self.dropout = nn.Dropout(config.dropout)

    @staticmethod
    def collect(
        agent_to_msg, # [batch, total_agents, n_emb]
        connections, # [batch, total_agents, local_agent_num]
        ):
        # to account for empty message [denoted as -1]
        connections = connections.long() + 1
        connections = connections.unsqueeze(-1).expand(-1, -1, -1, agent_to_msg.size(-1))  # [batch, block, local_agent_num, n_emb]
        # gather along dim=1 (number of agents)
        out = torch.gather(
            agent_to_msg.unsqueeze(2).expand(-1, -1, connections.size(2), -1),
            dim = 1,
            index = connections,
        )  
        return out # [batch, total_agents, local_agent_num, n_emb]
    
    def forward(self, agent_to_msg, connections):
        device = agent_to_msg.device
        b, c, _ = agent_to_msg.shape
        empty_msg = torch.zeros(b, 1, dtype=torch.long, device=device)
        empty_msg = self.empty_msg_emb(empty_msg)
        empty_msg = self.dropout(empty_msg)

        # add empty message vector for coordinator
        msg = torch.cat([empty_msg, agent_to_msg], dim=1) 

        # collect local messages for each agent from others
        msg = self.collect(msg, connections)
        return msg


# TODO: tranfer init weights function to higher class
# TODO: add message support
class RepresentationDecoder(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config
        self.n_head = config.n_head
        self.empty_connection_code = config.empty_connection_code

        self.transformer = nn.ModuleDict(dict(
            drop=nn.Dropout(config.dropout),
            act_msg_embd=nn.Embedding(1, config.action_msg_feats),
            h=nn.ModuleList([Block(config, config.latent_embd) for _ in range(config.n_decoder_layer)]),
            ln_f=LayerNorm(config.action_msg_feats, bias=config.bias),
            out_block=Block(config, 
                            config.action_msg_feats,
                            config.latent_embd,
                            config.n_embd
                            ),
        #    action_head = nn.Linear(config.action_msg_feats, 5, bias=False),
            msg_head = nn.Linear(config.action_msg_feats, config.latent_embd, bias=False)
        ))
        
        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_decoder_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, latent, messages, connections):
        device = latent.device
        b, t, h = latent.size()
        _, k = connections.size()
        
        # mask empty tokens out
        empty_mask_idx = (connections == self.empty_connection_code)
        empty_mask_idx = torch.cat(
            [torch.zeros(b, t, dtype=torch.bool, device=device),
             empty_mask_idx
             ],
             dim=1
        )
        attn_bias = torch.zeros(b, self.n_head, t+k, t+k, device=device)
        attn_bias = attn_bias.masked_fill(empty_mask_idx[:, None, None, :], float('-inf'))

        act_msg_latent = torch.arange(0, 1, dtype=torch.long, device=device)
        act_msg_latent = self.transformer.act_msg_embd(act_msg_latent)[None, :, :].repeat(b, 1, 1)
        act_msg_latent = self.transformer.drop(act_msg_latent)

        x = torch.cat([latent, messages], dim=1)

        for block in self.transformer.h:
            x = block(x, attn_bias=attn_bias)

        # empty mask for making output features
        empty_mask_latent = empty_mask_idx[:, None, :] # B , T --> B, 1, T
        attn_bias = torch.zeros_like(empty_mask_latent, dtype=torch.float, device=device)
        attn_bias = attn_bias.masked_fill(empty_mask_latent, float('-inf'))
        attn_bias = attn_bias[:, None, :, :].repeat(1, self.n_head, 1, 1)

        act_msg_latent = self.transformer.out_block(act_msg_latent, x, attn_bias = attn_bias)
        act_msg_latent = self.transformer.ln_f(act_msg_latent)

       # action = self.transformer.action_head(act_msg_latent)
#        action = act_msg_latent.squeeze(1)
 #       message = self.transformer.msg_head(act_msg_latent)
        return act_msg_latent


class DMM(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config
        self.n_comm_rounds = config.n_comm_rounds

        self.max_num_neighbors = config.max_num_neighbors
        self.agent_info_size = config.agent_info_size
        self.field_of_view_size = config.field_of_view_size
        self.block_size = config.block_size

        self.representation_encoder = RepresentationEncoder(self.config)
        self.representation_decoder = RepresentationDecoder(self.config) #NEW
        self.coordinator = MessageCoordinator(self.config)

        self.msg_nbrs_embedding = nn.Embedding(self.max_num_neighbors, config.latent_embd)
        # one linear head per predicted future action
        self.num_action_heads = getattr(config, "num_action_heads", 1)
        self.action_heads = nn.ModuleList(
            [nn.Linear(config.action_msg_feats, 5, bias=False) for _ in range(self.num_action_heads)]
        )
        # keep first head under old name for compatibility with existing code (e.g. RL wrapper)
        self.action_head = self.action_heads[0]
        
        # store loss weights from config, or default to equal weights
        if config.loss_weights is not None:
            assert len(config.loss_weights) == self.num_action_heads, \
                f"loss_weights length ({len(config.loss_weights)}) must match num_action_heads ({self.num_action_heads})"
            self.loss_weights = tuple(config.loss_weights)
        else:
            # default to equal weights
            self.loss_weights = tuple([1.0] * self.num_action_heads)
        #self.action_att = nn.Linear(config.action_msg_feats, 1, bias=False)
        # NEW weight tying
    #    self.representation_encoder.transformer.wte.weight = self.representation_decoder.action_head.weight

    def forward(self, observations, agent_chat_ids, target_actions=None):
        B, C, T = observations.shape
        device = observations.device
        _, _, L = agent_chat_ids.shape
        observations = observations.view(B * C, T)
        connections = agent_chat_ids.view(B * C, L)

        latent = self.representation_encoder(observations)
        agent_to_msg = torch.zeros(B*C, dtype=torch.long, device=device)
        agent_to_msg = self.coordinator.empty_msg_emb(agent_to_msg)
        agent_to_msg = self.coordinator.dropout(agent_to_msg)

        # create neighbor embeddings for messages
        nbrs = torch.arange(self.max_num_neighbors, device=device, dtype=torch.long)
        nbrs = self.msg_nbrs_embedding(nbrs)

        act_msg_latent = None
        for _ in range(self.n_comm_rounds):
            agent_to_msg = agent_to_msg.view(B, C, -1)
            messages = self.coordinator(agent_to_msg, agent_chat_ids)
            messages = messages.view(B * C, L, -1) + nbrs[:L]
            act_msg_latent = self.representation_decoder(latent, messages, connections)
            agent_to_msg = self.representation_decoder.transformer.msg_head(act_msg_latent)
      
        # attention pooling of features
        #action_features = torch.cat(action_features, 1)
      #  action_coofs = self.action_att(action_features)
      #  action_features = action_features*torch.softmax(action_coofs, dim=-1)
      #  action_features = action_features.sum(1)

      #  action_logits = self.action_head(action_features)

        # compute action logits and (optionally) multi-head loss
        # act_msg_latent: [B*C, 1, action_msg_feats]
        action_logits_first = self.action_heads[0](act_msg_latent.squeeze(1))

        if target_actions is not None:
            # target_actions expected shape: [B, C, horizon]
            if target_actions.dim() == 2:
                # fallback: [B, C] - single action per agent
                targets_flat = target_actions.reshape(-1)
                loss = F.cross_entropy(action_logits_first, targets_flat, ignore_index=-1)
                return action_logits_first, loss

            B_t, C_t, H = target_actions.shape
            assert B_t == B and C_t == C, "Target actions shape must be [B, C, H]"
            H_used = min(self.num_action_heads, H)

            # use weights from config
            weights = torch.tensor(self.loss_weights[:H_used], device=device, dtype=torch.float32)

            losses = []
            targets = target_actions[:, :, :H_used].reshape(B * C, H_used)
            for h_idx in range(H_used):
                head_logits = self.action_heads[h_idx](act_msg_latent.squeeze(1))
                head_targets = targets[:, h_idx]
                losses.append(F.cross_entropy(head_logits, head_targets, ignore_index=-1))

            weighted = torch.stack(losses) * weights
            loss = weighted.sum() / weights.sum()
        else:
            loss = None

        # For both training and inference, return logits of the first head
        return action_logits_first, loss

    def get_num_params(self, ):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        return n_params

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        logger.debug(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        logger.debug(
            f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        logger.debug(f"using fused AdamW: {use_fused}")

        return optimizer

    @torch.no_grad()
    def act(self, obs, agent_chat_ids, do_sample=True):
        logits, _ = self(obs, agent_chat_ids)

        probs = F.softmax(logits, dim=-1)

        if do_sample:
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            _, idx_next = torch.topk(probs, k=1, dim=-1)
        return idx_next.squeeze()
    

    def encode(self, observations):
        B, C, T = observations.shape
        device = observations.device
        observations = observations.view(B * C, T)

        latent = self.representation_encoder(observations)
        _, T_l, N_l = latent.shape
        return latent.view(B, C, T_l, N_l)



class DMM_RLWrapper(nn.Module):
    def __init__(self, model, config):
        super().__init__()
        self.dmm = model
     #   self.msg_action_head = nn.Linear(config.action_msg_feats, config.latent_embd*2+5, bias=False)
        self.msg_sigma_head = nn.Linear(config.action_msg_feats, config.latent_embd, bias=False)
        self.msg_mean_ln = LayerNorm(config.latent_embd)
        self.msg_sigma_ln = LayerNorm(config.latent_embd)
        self.config = config
        self.n_comm_rounds = config.n_comm_rounds
        self.num_actions = 5

        self.dmm.action_head.requires_grad = False

        self.max_num_neighbors = config.max_num_neighbors
        self.latent_embd = config.latent_embd

    @torch.no_grad()
    def act(self, obs, agent_chat_ids, do_sample=True):
        B, N_ag, N_nbrs = agent_chat_ids.shape
        device = agent_chat_ids.device
        latent = self.encode(obs)
        agent_to_msg = torch.zeros(B*N_ag, dtype=torch.long, device=device)
        agent_to_msg = self.dmm.coordinator.empty_msg_emb(agent_to_msg)

        for _ in range(self.n_comm_rounds):
            msg_mean, msg_sigma = self(latent, agent_to_msg, agent_chat_ids)
            msg_size = msg_sigma.shape[-1]
            b, _, _ = msg_sigma.shape

            msg_sigma = F.sigmoid(msg_sigma)
            msg_sigma = torch.clamp(msg_sigma, min=1e-3, max=1.0) 

            if do_sample:
                msg = torch.randn(b, 1, msg_size, device=device)
                msg = msg_mean  + msg * msg_sigma
            else:
                msg = msg_mean
            agent_to_msg = msg.view(b, msg_size)
        action_logits = agent_to_msg[:, :self.num_actions]
        probs = F.softmax(action_logits, dim=-1)

        if do_sample:
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            _, idx_next = torch.topk(probs, k=1, dim=-1)
        return idx_next.squeeze()
       # return torch.stack(actions).mode(0).values.squeeze()


    def encode(self, observations):
        return self.dmm.encode(observations)
    
    def get_empty_msg(self, empty):
        return self.dmm.coordinator.empty_msg_emb(empty)

    
    def forward(self, latent, agent_to_msg, agent_chat_ids):
        device = agent_chat_ids.device
        B, C, L = agent_chat_ids.shape
        _, _, T_l, N_l = latent.shape
        connections = agent_chat_ids.view(B * C, L)
        latent = latent.view(B * C, T_l, N_l)

        # create neighbor embeddings for messages
        nbrs = torch.arange(self.max_num_neighbors, device=device, dtype=torch.long)
        nbrs = self.dmm.msg_nbrs_embedding(nbrs)

        agent_to_msg = agent_to_msg.view(B, C, -1)
        messages = self.dmm.coordinator(agent_to_msg, agent_chat_ids)
        messages = messages.view(B * C, L, -1) + nbrs
        act_msg_latent = self.dmm.representation_decoder(
            latent, messages, connections
        )
        msg_mean = self.dmm.representation_decoder.transformer.msg_head(act_msg_latent)
        msg_sigma = self.msg_sigma_head(act_msg_latent)

        msg_mean = self.msg_mean_ln(msg_mean)
        msg_sigma = self.msg_sigma_ln(msg_sigma)

        return msg_mean, msg_sigma
    
    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        logger.debug(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        logger.debug(
            f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        logger.debug(f"using fused AdamW: {use_fused}")

        return optimizer
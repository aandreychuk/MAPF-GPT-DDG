import math
import inspect
from dataclasses import dataclass

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
    num_action_heads: int = 3  # Number of action predictions to make
    loss_weights: list = None  # Weights for each action head loss (e.g., [4, 2, 1] for 3 heads)



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
        nbrs = torch.arange(self.max_num_neighbors, device=device, dtype=torch.long)
        nbrs = self.transformer.wne(nbrs.repeat_interleave(self.agent_info_size))
        tail_size = self.block_size - self.field_of_view_size - self.max_num_neighbors*self.agent_info_size
        nbrs_embd = torch.cat(
            [
                torch.zeros(self.field_of_view_size, nbrs.shape[-1], device=device, dtype=nbrs.dtype),
                nbrs,
                torch.zeros(tail_size, nbrs.shape[-1], device=device, dtype=nbrs.dtype)
            ],
            dim = 0
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


# TODO: tranfer init weights function to higher class
class RepresentationDecoder(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config
        self.n_head = config.n_head

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

    def forward(self, latent):
        device = latent.device
        b, t, h = latent.size()
        
        act_msg_latent = torch.arange(0, 1, dtype=torch.long, device=device)
        act_msg_latent = self.transformer.act_msg_embd(act_msg_latent)[None, :, :].repeat(b, 1, 1)
        act_msg_latent = self.transformer.drop(act_msg_latent)

        x = latent

        for block in self.transformer.h:
            x = block(x, attn_bias=None)

        act_msg_latent = self.transformer.out_block(act_msg_latent, x, attn_bias=None)
        act_msg_latent = self.transformer.ln_f(act_msg_latent)

        return act_msg_latent


class DMM(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.max_num_neighbors = config.max_num_neighbors
        self.agent_info_size = config.agent_info_size
        self.field_of_view_size = config.field_of_view_size
        self.block_size = config.block_size

        self.representation_encoder = RepresentationEncoder(self.config)
        self.representation_decoder = RepresentationDecoder(self.config)
        
        # Create multiple action heads
        self.num_action_heads = config.num_action_heads
        self.action_heads = nn.ModuleList([
            nn.Linear(config.action_msg_feats, 5, bias=False) 
            for _ in range(config.num_action_heads)
        ])
        
        # Set up loss weights (default: equal weights, or use provided weights)
        if config.loss_weights is None:
            self.loss_weights = [1.0] * config.num_action_heads
        else:
            assert len(config.loss_weights) == config.num_action_heads, \
                f"Number of loss weights ({len(config.loss_weights)}) must match num_action_heads ({config.num_action_heads})"
            self.loss_weights = config.loss_weights

    def forward(self, observations, target_actions=None):
        B, T = observations.shape
        
        latent = self.representation_encoder(observations)
        act_msg_latent = self.representation_decoder(latent)
        decoder_output = act_msg_latent.squeeze(1)  # (B, action_msg_feats)
        
        # Generate predictions from all action heads
        action_logits_list = []
        for action_head in self.action_heads:
            action_logits_list.append(action_head(decoder_output))
        
        # Stack to shape (B, num_action_heads, 5) or (B, 5) if single head
        if self.num_action_heads == 1:
            action_logits = action_logits_list[0]
        else:
            action_logits = torch.stack(action_logits_list, dim=1)  # (B, num_action_heads, 5)
        
        if target_actions is not None:
            # Handle target_actions: (B, num_action_heads) or (B, action_horizon)
            if target_actions.dim() == 2:
                # If shape is (B, action_horizon), we expect it to match num_action_heads
                if target_actions.size(-1) != self.num_action_heads:
                    raise ValueError(
                        f"target_actions last dimension {target_actions.size(-1)} != num_action_heads {self.num_action_heads}"
                    )
            elif target_actions.dim() == 1:
                # Single action per sample - expand to match num_action_heads
                target_actions = target_actions.unsqueeze(1).expand(-1, self.num_action_heads)
            
            assert target_actions.shape == (B, self.num_action_heads), \
                f"target_actions shape {target_actions.shape} != (B={B}, num_action_heads={self.num_action_heads})"
            
            # Compute weighted loss for each action head
            losses = []
            for i in range(self.num_action_heads):
                head_logits = action_logits[:, i, :] if self.num_action_heads > 1 else action_logits
                head_targets = target_actions[:, i]
                head_loss = F.cross_entropy(head_logits, head_targets, ignore_index=-1)
                losses.append(self.loss_weights[i] * head_loss)
            
            loss = sum(losses)/sum(self.loss_weights)
        else:
            loss = None

        return action_logits, loss

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
    def act(self, obs, do_sample=True):
        logits, _ = self(obs)

        # During inference, only use the first action head
        if self.num_action_heads > 1:
            # logits shape: (B, num_action_heads, 5)
            first_head_logits = logits[:, 0, :]  # (B, 5)
        else:
            # logits shape: (B, 5)
            first_head_logits = logits

        probs = F.softmax(first_head_logits, dim=-1)

        if do_sample:
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            _, idx_next = torch.topk(probs, k=1, dim=-1)
        return idx_next.squeeze()
    

    def encode(self, observations):
        B, T = observations.shape
        device = observations.device

        latent = self.representation_encoder(observations)
        _, T_l, N_l = latent.shape
        return latent.view(B, 1, T_l, N_l)



class DMM_RLWrapper(nn.Module):
    def __init__(self, model, config):
        super().__init__()
        self.dmm = model
        self.mean_proj = nn.Linear(config.action_msg_feats, config.latent_embd, bias=False)
        self.sigma_proj = nn.Linear(config.action_msg_feats, config.latent_embd, bias=False)
        self.mean_ln = LayerNorm(config.latent_embd)
        self.sigma_ln = LayerNorm(config.latent_embd)
        self.config = config
        self.num_actions = 5

        self.dmm.action_head.requires_grad = False

        self.latent_embd = config.latent_embd

    @torch.no_grad()
    def act(self, obs, do_sample=True):
        device = obs.device
        latent = self.encode(obs)
        
        mean, sigma = self(latent)
        latent_size = sigma.shape[-1]
        b, _ = sigma.shape

        sigma = F.sigmoid(sigma)
        sigma = torch.clamp(sigma, min=1e-3, max=1.0) 

        if do_sample:
            noise = torch.randn(b, latent_size, device=device)
            action_latent = mean + noise * sigma
        else:
            action_latent = mean
        action_logits = action_latent[:, :self.num_actions]
        
        probs = F.softmax(action_logits, dim=-1)

        if do_sample:
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            _, idx_next = torch.topk(probs, k=1, dim=-1)
        return idx_next.squeeze()


    def encode(self, observations):
        return self.dmm.encode(observations)
    
    def forward(self, latent):
        _, _, T_l, N_l = latent.shape
        latent = latent.view(-1, T_l, N_l)

        decoder_output = self.dmm.representation_decoder(latent)
        
        mean = self.mean_proj(decoder_output).squeeze(1)
        sigma = self.sigma_proj(decoder_output).squeeze(1)

        mean = self.mean_ln(mean)
        sigma = self.sigma_ln(sigma)

        return mean, sigma
    
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



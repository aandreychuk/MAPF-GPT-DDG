import math
import inspect
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch.nn import functional as F
from loguru import logger

try:
    from torch_geometric.nn import TransformerConv
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False
    logger.warning("torch_geometric not available. Install with: pip install torch-geometric")


class LayerNorm(nn.Module):
    """RMSNorm for consistency with original architecture."""
    def __init__(self, ndim, bias=None):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))

    def forward(self, input):
        return F.rms_norm(input, self.weight.shape, self.weight, 1e-5)


class ResNetBlock(nn.Module):
    """ResNet-style residual block for spatial encoding."""
    def __init__(self, channels, bias=False):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=bias)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=bias)
        self.activation = nn.GELU()
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.activation(out)
        out = self.conv2(out)
        out = out + residual  # Residual connection
        out = self.activation(out)
        return out


class SpatialEncoder(nn.Module):
    """
    Encodes 11x11 spatial FOV observations using ResNet-style CNN.
    Processes token sequences as spatial grids with hierarchical feature learning.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.n_embd = config.n_embd
        self.field_of_view_size = config.field_of_view_size
        self.grid_size = int(math.sqrt(self.field_of_view_size))
        
        # Embedding for tokens
        self.token_embedding = nn.Embedding(self.vocab_size, self.n_embd)
        
        # Initial projection to spatial features
        self.initial_conv = nn.Conv2d(
            self.n_embd, self.n_embd, 
            kernel_size=3, padding=1, bias=config.bias
        )
        
        # ResNet blocks for hierarchical feature learning
        n_resnet_blocks = getattr(config, 'n_resnet_blocks', 2)
        self.resnet_blocks = nn.ModuleList([
            ResNetBlock(self.n_embd, bias=config.bias)
            for _ in range(n_resnet_blocks)
        ])
        
        # Global pooling
        self.pool = nn.AdaptiveAvgPool2d(1)  # Global average pooling
        
        # Project to latent dimension
        self.latent_proj = nn.Linear(self.n_embd, config.latent_embd, bias=config.bias)
        self.ln = LayerNorm(config.latent_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, observations):
        """
        Args:
            observations: [B*C, T] token sequence
        Returns:
            latent: [B*C, latent_embd] encoded representation
        """
        B_C, _ = observations.shape
        
        # Embed tokens
        tok_emb = self.token_embedding(observations)  # [B*C, T, n_embd]
        
        # Reshape to [B*C, n_embd, 11, 11]
        spatial = tok_emb.transpose(1, 2)  # [B*C, n_embd, T]
        spatial = spatial.view(B_C, self.n_embd, self.grid_size, self.grid_size)
        
        # ResNet blocks for hierarchical features
        for resnet_block in self.resnet_blocks:
            spatial = resnet_block(spatial)
        
        # Global pooling
        spatial = self.pool(spatial).squeeze(-1).squeeze(-1)  # [B*C, n_embd]
        
        # Project to latent
        latent = self.latent_proj(spatial)  # [B*C, latent_embd]
        latent = self.ln(latent)
        latent = self.dropout(latent)
        
        return latent


class GraphTransformerLayer(nn.Module):
    """
    Graph Transformer layer using TransformerConv (latest GNN architecture).
    Combines graph attention with transformer-style processing.
    """
    def __init__(self, in_channels, out_channels, num_heads=4, dropout=0.0, bias=False, use_edge_attr=False, edge_dim=2):
        super().__init__()
        if not TORCH_GEOMETRIC_AVAILABLE:
            raise ImportError("torch_geometric is required for GNN architecture")
        
        self.use_edge_attr = use_edge_attr
        
        # Edge embedding if using edge attributes
        if use_edge_attr:
            # Project edge attributes (e.g., 2D coords) to feature dimension
            self.edge_embedding = nn.Linear(edge_dim, in_channels, bias=bias)
            edge_dim_for_conv = in_channels
        else:
            self.edge_embedding = None
            edge_dim_for_conv = None
            
        # Use concat=True for multi-head attention (standard practice)
        # This outputs num_heads * out_channels, which we'll project back to out_channels
        self.conv = TransformerConv(
            in_channels=in_channels,
            out_channels=out_channels,
            heads=num_heads,
            dropout=dropout,
            edge_dim=edge_dim_for_conv,
            bias=bias,
            concat=True  # Always concat for multi-head attention
        )
        
        # When concat=True, TransformerConv outputs [N, num_heads * out_channels]
        # We need to project it back to [N, out_channels]
        self.out_proj = nn.Linear(num_heads * out_channels, out_channels, bias=bias)
            
        self.ln = LayerNorm(out_channels, bias=bias)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, edge_index, edge_attr=None):
        """
        Args:
            x: [N, in_channels] node features
            edge_index: [2, E] edge indices
            edge_attr: [E, edge_dim] optional edge attributes (e.g., [E, 2] for dx, dy)
        Returns:
            out: [N, out_channels] updated node features
        """
        # Embed edge attributes if provided and enabled
        if self.use_edge_attr and edge_attr is not None:
            edge_attr_embedded = self.edge_embedding(edge_attr)  # [E, edge_dim] -> [E, in_channels]
        else:
            edge_attr_embedded = None
            
        # Graph transformer convolution
        # When concat=True, this outputs [N, num_heads * out_channels]
        out = self.conv(x, edge_index, edge_attr_embedded)
        
        # Project from [N, num_heads * out_channels] to [N, out_channels]
        out = self.out_proj(out)
        
        # Residual connection and normalization
        if out.size(-1) == x.size(-1):
            out = out + x
        out = self.ln(out)
        out = self.dropout(out)
        
        return out


class GraphMLP(nn.Module):
    """MLP for graph nodes."""
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.0, bias=False):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_channels, bias=bias),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, out_channels, bias=bias),
            nn.Dropout(dropout)
        )
        self.ln = LayerNorm(out_channels, bias=bias)
        
    def forward(self, x):
        out = self.mlp(x)
        if out.size(-1) == x.size(-1):
            out = out + x  # Residual
        out = self.ln(out)
        return out


def build_agent_graph(agent_chat_ids, agents_rel_coords=None, device=None):
    """
    Build graph from agent connections (OPTIMIZED VERSION).
    Uses vectorized tensor operations instead of Python loops.
    
    Args:
        agent_chat_ids: [B, C, L] agent connection indices
        agents_rel_coords: [B, C, L*2] optional relative coordinates
        device: device for tensors
    Returns:
        edge_index: [2, E] edge indices
        edge_attr: [E, edge_dim] optional edge attributes
        batch: [N] batch assignment for nodes
    """
    B, C, L = agent_chat_ids.shape
    if device is None:
        device = agent_chat_ids.device
    
    B_C = B * C
    
    # Flatten batch and agents
    agent_chat_ids_flat = agent_chat_ids.view(B_C, L)  # [B*C, L]
    
    # Create agent indices: [0, 1, 2, ..., B*C-1]
    agent_ids = torch.arange(B_C, dtype=torch.long, device=device)  # [B*C]
    batch_ids = agent_ids // C  # [B*C] - batch index for each agent
    
    # ===== OPTIMIZED EDGE CONSTRUCTION =====
    # Self-loops: every agent connects to itself
    self_loop_src = agent_ids  # [B*C]
    self_loop_dst = agent_ids  # [B*C]
    
    # Neighbor edges: vectorized construction
    # Expand agent_ids to match neighbor slots: [B*C, L]
    src_agents = agent_ids.unsqueeze(1).expand(-1, L)  # [B*C, L]
    
    # Get neighbor local indices: [B*C, L]
    neighbor_local = agent_chat_ids_flat  # [B*C, L]
    
    # Create mask for valid neighbors (not -1 and within range)
    valid_mask = (neighbor_local >= 0) & (neighbor_local < C)  # [B*C, L]
    
    # Convert local neighbor indices to global agent indices
    # For each agent, compute: batch_id * C + neighbor_local_id
    batch_ids_expanded = batch_ids.unsqueeze(1).expand(-1, L)  # [B*C, L]
    neighbor_global = batch_ids_expanded * C + neighbor_local  # [B*C, L]
    
    # Apply valid mask
    valid_src = src_agents[valid_mask]  # [E_neighbors]
    valid_dst = neighbor_global[valid_mask]  # [E_neighbors]
    
    # Combine self-loops and neighbor edges
    edge_src = torch.cat([self_loop_src, valid_src], dim=0)  # [B*C + E_neighbors]
    edge_dst = torch.cat([self_loop_dst, valid_dst], dim=0)  # [B*C + E_neighbors]
    
    # Create edge_index: [2, E]
    edge_index = torch.stack([edge_src, edge_dst], dim=0)  # [2, E]
    
    # Batch assignment: same as agent batch_ids
    batch = batch_ids  # [B*C]
    
    # ===== OPTIMIZED EDGE ATTRIBUTE CONSTRUCTION =====
    edge_attr = None
    if agents_rel_coords is not None:
        # Reshape coordinates: [B, C, L*2] -> [B*C, L, 2]
        coords_flat = agents_rel_coords.view(B_C, L, 2)  # [B*C, L, 2]
        
        # Self-loop edge attributes: zeros for all agents
        self_loop_attr = torch.zeros(B_C, 2, device=device)  # [B*C, 2]
        
        # Neighbor edge attributes: use valid_mask to select coordinates
        neighbor_attr = coords_flat[valid_mask]  # [E_neighbors, 2]
        
        # Combine self-loop and neighbor attributes
        edge_attr = torch.cat([self_loop_attr, neighbor_attr], dim=0)  # [B*C + E_neighbors, 2]
    
    return edge_index, edge_attr, batch


class GraphEncoder(nn.Module):
    """
    Graph encoder that processes agent observations and builds agent graph.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.latent_embd = config.latent_embd
        self.n_gnn_layers = getattr(config, 'n_gnn_layers', 2)
        self.n_heads = getattr(config, 'gnn_heads', 4)
        
        # Spatial encoder for FOV
        self.spatial_encoder = SpatialEncoder(config)
        
        # Graph transformer layers
        # Enable edge attributes if relative coordinates are typically provided
        use_edge_attr = getattr(config, 'use_edge_attributes', True)
        self.gnn_layers = nn.ModuleList([
            GraphTransformerLayer(
                in_channels=self.latent_embd,
                out_channels=self.latent_embd,
                num_heads=self.n_heads,
                dropout=config.dropout,
                bias=config.bias,
                use_edge_attr=use_edge_attr,
                edge_dim=2  # dx, dy coordinates
            ) for _ in range(self.n_gnn_layers)
        ])
        
        # MLP layers
        self.mlp_layers = nn.ModuleList([
            GraphMLP(
                in_channels=self.latent_embd,
                hidden_channels=self.latent_embd * 2,
                out_channels=self.latent_embd,
                dropout=config.dropout,
                bias=config.bias
            ) for _ in range(self.n_gnn_layers)
        ])
    
    
    def forward(self, observations, agent_chat_ids, agents_rel_coords=None):
        """
        Args:
            observations: [B, C, T] token sequences
            agent_chat_ids: [B, C, L] agent connections
            agents_rel_coords: [B, C, L*2] optional relative coordinates
        Returns:
            node_features: [B*C, latent_embd] encoded agent features
        """
        B, C, T = observations.shape
        observations_flat = observations.view(B * C, T)
        
        # Encode spatial observations
        node_features = self.spatial_encoder(observations_flat)  # [B*C, latent_embd]
        
        # Build graph
        edge_index, edge_attr, batch = build_agent_graph(
            agent_chat_ids, agents_rel_coords, device=observations.device
        )
        
        # Apply graph transformer layers
        for gnn_layer, mlp_layer in zip(self.gnn_layers, self.mlp_layers):
            # Graph attention
            node_features = gnn_layer(node_features, edge_index, edge_attr)
            # MLP
            node_features = mlp_layer(node_features)
        
        return node_features


class GraphDecoder(nn.Module):
    """
    Graph decoder that processes agent features and messages to predict actions.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.latent_embd = config.latent_embd
        self.action_msg_feats = config.action_msg_feats
        self.n_comm_rounds = config.n_comm_rounds
        
        # Message processing layers
        use_edge_attr = getattr(config, 'use_edge_attributes', True)
        self.message_layers = nn.ModuleList([
            GraphTransformerLayer(
                in_channels=self.latent_embd,
                out_channels=self.latent_embd,
                num_heads=getattr(config, 'gnn_heads', 4),
                dropout=config.dropout,
                bias=config.bias,
                use_edge_attr=use_edge_attr,
                edge_dim=2  # dx, dy coordinates
            ) for _ in range(self.n_comm_rounds)
        ])
        
        # Action prediction head
        self.action_proj = nn.Linear(self.latent_embd, self.action_msg_feats, bias=config.bias)
        self.ln = LayerNorm(self.action_msg_feats, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, node_features, agent_chat_ids, agents_rel_coords=None):
        """
        Args:
            node_features: [B*C, latent_embd] agent features
            agent_chat_ids: [B, C, L] agent connections
            agents_rel_coords: [B, C, L*2] optional relative coordinates
        Returns:
            action_features: [B*C, action_msg_feats] action prediction features
        """
        device = node_features.device
        
        # Build graph once (structure doesn't change across communication rounds)
        # This is a major optimization - graph structure is the same, only node features change
        edge_index, edge_attr, batch = build_agent_graph(
            agent_chat_ids, agents_rel_coords, device=device
        )
        
        # Multiple communication rounds (reuse same graph structure)
        for message_layer in self.message_layers:
            # Message passing (graph structure is the same, only node features update)
            node_features = message_layer(node_features, edge_index, edge_attr)
        
        # Project to action features
        action_features = self.action_proj(node_features)
        action_features = self.ln(action_features)
        action_features = self.dropout(action_features)
        
        return action_features


@dataclass
class DMMGNNConfig:
    """Configuration for GNN-based DMM model."""
    block_size: int = 128
    vocab_size: int = 97
    field_of_view_size: int = 11*11
    
    # GNN-specific
    n_gnn_layers: int = 2  # Number of GNN layers in encoder
    gnn_heads: int = 2  # Number of attention heads in graph transformer
    n_resnet_blocks: int = 2  # Number of ResNet blocks in spatial encoder
    
    # Shared with DMM
    n_embd: int = 16
    latent_embd: int = 8
    dropout: float = 0.0
    bias: bool = False
    empty_token_code: int = 0
    action_msg_feats: int = 16
    empty_connection_code: int = -1
    n_comm_rounds: int = 2
    num_action_heads: int = 1
    loss_weights: Optional[tuple] = None


class DMMGNN(nn.Module):
    """
    GNN-based model for decentralized MAPF.
    Uses Graph Transformer Networks for agent communication.
    """
    def __init__(self, config):
        super().__init__()
        if not TORCH_GEOMETRIC_AVAILABLE:
            raise ImportError("torch_geometric is required. Install with: pip install torch-geometric")
            
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config
        self.n_comm_rounds = config.n_comm_rounds
        
        self.field_of_view_size = config.field_of_view_size
        self.block_size = config.block_size
        
        # Graph encoder (spatial + GNN)
        self.graph_encoder = GraphEncoder(config)
        
        # Graph decoder (message passing + action prediction)
        self.graph_decoder = GraphDecoder(config)
        
        # Action heads
        self.num_action_heads = getattr(config, "num_action_heads", 1)
        self.action_heads = nn.ModuleList(
            [nn.Linear(config.action_msg_feats, 5, bias=False) for _ in range(self.num_action_heads)]
        )
        self.action_head = self.action_heads[0]
        
        # Loss weights
        if config.loss_weights is not None:
            assert len(config.loss_weights) == self.num_action_heads, \
                f"loss_weights length ({len(config.loss_weights)}) must match num_action_heads ({self.num_action_heads})"
            self.loss_weights = tuple(config.loss_weights)
        else:
            self.loss_weights = tuple([1.0] * self.num_action_heads)
    
    def forward(self, observations, agent_chat_ids, target_actions=None, agents_rel_coords=None):
        """
        Args:
            observations: [B, C, T] token sequences
            agent_chat_ids: [B, C, L] agent connection indices
            target_actions: [B, C, H] optional target actions for loss
            agents_rel_coords: [B, C, L*2] optional relative coordinates
        Returns:
            action_logits: [B*C, 5] action logits
            loss: scalar loss if target_actions provided, else None
        """
        B, C, T = observations.shape
        device = observations.device
        
        # Encode observations and build agent graph
        node_features = self.graph_encoder(observations, agent_chat_ids, agents_rel_coords)
        
        # Decode to action features
        action_features = self.graph_decoder(node_features, agent_chat_ids, agents_rel_coords)
        
        # Predict actions
        action_logits_first = self.action_heads[0](action_features)
        
        if target_actions is not None:
            if target_actions.dim() == 2:
                targets_flat = target_actions.reshape(-1)
                loss = F.cross_entropy(action_logits_first, targets_flat, ignore_index=-1)
                return action_logits_first, loss
            
            B_t, C_t, H = target_actions.shape
            assert B_t == B and C_t == C, "Target actions shape must be [B, C, H]"
            H_used = min(self.num_action_heads, H)
            
            weights = torch.tensor(self.loss_weights[:H_used], device=device, dtype=torch.float32)
            losses = []
            targets = target_actions[:, :, :H_used].reshape(B * C, H_used)
            
            for h_idx in range(H_used):
                head_logits = self.action_heads[h_idx](action_features)
                head_targets = targets[:, h_idx]
                losses.append(F.cross_entropy(head_logits, head_targets, ignore_index=-1))
            
            weighted = torch.stack(losses) * weights
            loss = weighted.sum() / weights.sum()
        else:
            loss = None
        
        return action_logits_first, loss
    
    def get_num_params(self):
        """Return the number of parameters in the model."""
        n_params = sum(p.numel() for p in self.parameters())
        return n_params
    
    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        """Configure optimizer (same interface as DMM)."""
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        logger.debug(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        logger.debug(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        logger.debug(f"using fused AdamW: {use_fused}")
        
        return optimizer
    
    @torch.no_grad()
    def act(self, obs, agent_chat_ids, do_sample=True, agents_rel_coords=None):
        """Inference method (same interface as DMM)."""
        logits, _ = self(obs, agent_chat_ids, agents_rel_coords=agents_rel_coords)
        probs = F.softmax(logits, dim=-1)
        
        if do_sample:
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            _, idx_next = torch.topk(probs, k=1, dim=-1)
        return idx_next.squeeze()
    
    @torch.no_grad()
    def get_action_probs(self, obs, agent_chat_ids, agents_rel_coords=None):
        """Get action logits for the given observations and agent connections."""
        logits, _ = self(obs, agent_chat_ids, agents_rel_coords=agents_rel_coords)
        return F.softmax(logits, dim=-1)


from __future__ import annotations
import math
import torch
from torch import nn
from torch.nn import functional as F
CUTOFF=6.0
RBF_COUNT=24
def cosine_cutoff(dist: torch.Tensor, cutoff: float = CUTOFF) -> torch.Tensor:
    """Smooth cosine cutoff envelope going smoothly to 0 at cutoff."""
    cut = 0.5 * (torch.cos(torch.clamp(dist, max=cutoff) * (math.pi / cutoff)) + 1.0)
    return torch.where(dist <= cutoff, cut, torch.zeros_like(cut))

def rbf_features(dist: torch.Tensor, cutoff: float = CUTOFF, num_rbf: int = RBF_COUNT) -> torch.Tensor:
    """Gaussian Radial Basis Function expansions."""
    centers = torch.linspace(0.0, cutoff, num_rbf, device=dist.device)
    widths = torch.full_like(centers, (cutoff / num_rbf))
    # Gaussian expansion: exp(-0.5 * ((d - mu) / sigma)^2)
    diff = (dist[..., None] - centers) / widths
    return torch.exp(-0.5 * diff.square())

class PaiNNMessageV2(nn.Module):
    """Full PaiNN message layer supporting scalar-vector and vector-vector message passing."""

    def __init__(self, hidden_dim: int, num_rbf: int = RBF_COUNT, num_edge_types: int = 3):
        super().__init__()
        self.hidden_dim = hidden_dim
        # Edge type embedding: 0=intra-solute, 1=intra-solvent, 2=inter-molecular
        self.edge_emb = nn.Embedding(num_edge_types, 16)
        
        # Scalar filter mapping (RBF + edge_emb) -> 3 * hidden_dim
        # Splits into: phi_ss (scalar->scalar), phi_sv (vector->scalar), phi_vs (scalar->vector)
        self.scalar_filter = nn.Sequential(
            nn.Linear(num_rbf + 16, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3 * hidden_dim)
        )
        
        # Vector-to-vector filter mapping -> hidden_dim
        self.vec_filter = nn.Sequential(
            nn.Linear(num_rbf + 16, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.scalar_proj = nn.Linear(hidden_dim, 3 * hidden_dim)

    def forward(
        self,
        h: torch.Tensor,          # (B, N, F)
        vec: torch.Tensor,        # (B, N, F, 3)
        unit_dir: torch.Tensor,   # (B, N, N, 3)
        rbf: torch.Tensor,        # (B, N, N, num_rbf)
        edge_type: torch.Tensor,  # (B, N, N)
        valid: torch.Tensor,      # (B, N, N) bool mask
        cut: torch.Tensor,        # (B, N, N)
        edge_gate: torch.Tensor | None = None,  # (B, N, N), optional attribution gate
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, N, F = h.shape
        
        # Combine RBF with edge type embedding
        edge_feat = torch.cat([rbf, self.edge_emb(edge_type)], dim=-1)  # (B, N, N, num_rbf + 16)
        
        # Filter weights
        phi = self.scalar_filter(edge_feat)  # (B, N, N, 3 * F)
        phi_ss, phi_sv, phi_vs = phi.chunk(3, dim=-1)
        
        phi_vv = self.vec_filter(edge_feat)  # (B, N, N, F)
        
        # Mask and cutoff envelope
        envelope = (cut * valid.float())[..., None]  # (B, N, N, 1)
        if edge_gate is not None:
            envelope = envelope * edge_gate[..., None]
        phi_ss = phi_ss * envelope
        phi_sv = phi_sv * envelope
        phi_vs = phi_vs * envelope
        phi_vv = phi_vv * envelope
        
        # Project source scalar states: h_j -> (s_ss, s_sv, s_vs)
        s_proj = self.scalar_proj(h)  # (B, N, 3 * F)
        s_ss, s_sv, s_vs = s_proj.chunk(3, dim=-1)
        
        # 1. Scalar-to-scalar contribution: s_ss_j * phi_ss_ij
        msg_ss = s_ss[:, None, :, :] * phi_ss  # (B, N, N, F)
        
        # 2. Vector-to-scalar contribution: <v_j, unit_dir_ij> * phi_sv_ij * s_sv_j
        # vec is (B, N, F, 3), unit_dir is (B, N_i, N_j, 3)
        # v_j dot unit_dir_ij -> (B, N_i, N_j, F)
        v_dot_dir = (vec[:, None, :, :, :] * unit_dir[:, :, :, None, :]).sum(dim=-1)
        msg_sv = v_dot_dir * phi_sv * s_sv[:, None, :, :]
        
        d_scalar = (msg_ss + msg_sv).sum(dim=2)  # sum over j -> (B, N, F)
        
        # 3. Scalar-to-vector contribution: s_vs_j * phi_vs_ij * unit_dir_ij
        msg_vs = (s_vs[:, None, :, :] * phi_vs)[..., None] * unit_dir[:, :, :, None, :]  # (B, N, N, F, 3)
        
        # 4. Vector-to-vector contribution: v_j * phi_vv_ij
        msg_vv = vec[:, None, :, :, :] * phi_vv[..., None]  # (B, N, N, F, 3)
        
        d_vec = (msg_vs + msg_vv).sum(dim=2)  # sum over j -> (B, N, F, 3)
        
        return d_scalar, d_vec

class PaiNNUpdateV2(nn.Module):
    """PaiNN update layer: equivariant coupling between scalar and vector features."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.u_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim * 3)
        )

    def forward(self, h: torch.Tensor, vec: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Linear projections for vectors (along feature dim F): (B, N, F, 3)
        u = self.u_proj(vec.transpose(-1, -2)).transpose(-1, -2)  # (B, N, F, 3)
        v = self.v_proj(vec.transpose(-1, -2)).transpose(-1, -2)  # (B, N, F, 3)
        
        v_norm = torch.linalg.norm(v, dim=-1)  # (B, N, F)
        
        # Concatenate scalar feature with norm of projected vector
        mlp_in = torch.cat([h, v_norm], dim=-1)  # (B, N, 2 * F)
        a_s, a_vv, a_sv = self.mlp(mlp_in).chunk(3, dim=-1)
        
        # Scalar update: a_s + a_sv * <u, v>
        u_dot_v = (u * v).sum(dim=-1)  # (B, N, F)
        d_scalar = a_s + a_sv * u_dot_v
        
        # Vector update: u * a_vv
        d_vec = u * a_vv[..., None]
        
        return d_scalar, d_vec

class SolvationReadoutHead(nn.Module):
    """Separate Solute & Solvent asymmetric interaction pooling and descriptor late fusion."""

    def __init__(self, hidden_dim: int, desc_dim: int = 16, head_hidden: int = 128):
        super().__init__()
        in_dim = hidden_dim * 6
        
        self.desc_net = nn.Sequential(
            nn.Linear(desc_dim, 32),
            nn.SiLU(),
            nn.LayerNorm(32)
        )
        
        self.interaction_head = nn.Sequential(
            nn.Linear(in_dim + 32, head_hidden),
            nn.SiLU(),
            nn.Dropout(0.05),
            nn.Linear(head_hidden, head_hidden // 2),
            nn.SiLU(),
            nn.Linear(head_hidden // 2, 1)
        )

    def forward(
        self,
        h: torch.Tensor,        # (B, N, F)
        vec: torch.Tensor,      # (B, N, F, 3)
        role: torch.Tensor,     # (B, N) 0: solute, 1: solvent
        mask: torch.Tensor,     # (B, N) bool
        desc: torch.Tensor      # (B, desc_dim)
    ) -> torch.Tensor:
        solute_mask = mask & (role == 0)  # (B, N)
        solvent_mask = mask & (role == 1) # (B, N)
        
        s_denom = solute_mask.sum(dim=1, keepdim=True).clamp(min=1).unsqueeze(-1)   # (B, 1, 1)
        w_denom = solvent_mask.sum(dim=1, keepdim=True).clamp(min=1).unsqueeze(-1)  # (B, 1, 1)
        
        h_solute = (h * solute_mask[..., None].float()).sum(dim=1, keepdim=True) / s_denom    # (B, 1, F)
        h_solvent = (h * solvent_mask[..., None].float()).sum(dim=1, keepdim=True) / w_denom  # (B, 1, F)
        
        h_solute = h_solute.squeeze(1)   # (B, F)
        h_solvent = h_solvent.squeeze(1) # (B, F)
        
        v_norm = torch.linalg.norm(vec, dim=-1)  # (B, N, F)
        v_solute = ((v_norm * solute_mask[..., None].float()).sum(dim=1, keepdim=True) / s_denom).squeeze(1)
        v_solvent = ((v_norm * solvent_mask[..., None].float()).sum(dim=1, keepdim=True) / w_denom).squeeze(1)
        
        h_prod = h_solute * h_solvent
        h_diff = torch.abs(h_solute - h_solvent)
        
        features = torch.cat([h_solute, h_solvent, h_prod, h_diff, v_solute, v_solvent], dim=-1)
        desc_feat = self.desc_net(desc)
        
        combined = torch.cat([features, desc_feat], dim=-1)
        return self.interaction_head(combined).squeeze(-1)

class PaiNNModelV2(nn.Module):
    """PaiNN Model for Solvation Free Energy Prediction."""

    def __init__(self, hidden_dim: int = 96, num_layers: int = 3, desc_dim: int = 16):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.atom_emb = nn.Embedding(100, hidden_dim, padding_idx=0)
        self.role_emb = nn.Embedding(2, hidden_dim)
        
        self.messages = nn.ModuleList([
            PaiNNMessageV2(hidden_dim=hidden_dim, num_rbf=RBF_COUNT, num_edge_types=3)
            for _ in range(num_layers)
        ])
        self.updates = nn.ModuleList([
            PaiNNUpdateV2(hidden_dim=hidden_dim)
            for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim)
            for _ in range(num_layers)
        ])
        
        self.head = SolvationReadoutHead(hidden_dim=hidden_dim, desc_dim=desc_dim)

    def forward(
        self,
        z: torch.Tensor,        # (B, N)
        pos: torch.Tensor,      # (B, N, 3)
        role: torch.Tensor,     # (B, N)
        mask: torch.Tensor,     # (B, N)
        desc: torch.Tensor,     # (B, desc_dim)
        edge_gate: torch.Tensor | None = None,  # optional edge gate for attribution
    ) -> torch.Tensor:
        # 1. Compute geometric relations
        rel = pos[:, :, None, :] - pos[:, None, :, :]  # r_ij = p_j - p_i -> (B, N, N, 3)
        dist = torch.linalg.norm(rel + 1e-12, dim=-1)   # (B, N, N)
        unit_dir = rel / dist[..., None].clamp(min=1e-6) # unit direction (B, N, N, 3)
        
        valid = mask[:, :, None] & mask[:, None, :] & (dist < CUTOFF) & (dist > 1e-3)
        rbf = rbf_features(dist, cutoff=CUTOFF, num_rbf=RBF_COUNT)
        cut = cosine_cutoff(dist, cutoff=CUTOFF)
        
        # 2. Compute edge types:
        r_i = role[:, :, None]
        r_j = role[:, None, :]
        edge_type = torch.where(
            (r_i == 0) & (r_j == 0),
            torch.zeros_like(r_i.expand_as(dist)),
            torch.where(
                (r_i == 1) & (r_j == 1),
                torch.ones_like(r_i.expand_as(dist)),
                torch.full_like(r_i.expand_as(dist), 2)
            )
        )
        
        # 3. Initial embeddings
        h = self.atom_emb(z) + self.role_emb(role)  # (B, N, F)
        vec = torch.zeros((*h.shape, 3), device=h.device)  # (B, N, F, 3)
        
        # 4. Message passing layers
        for msg_layer, upd_layer, norm_layer in zip(self.messages, self.updates, self.norms):
            ds, dv = msg_layer(h, vec, unit_dir, rbf, edge_type, valid, cut, edge_gate=edge_gate)
            h = h + ds
            vec = vec + dv
            
            ds_upd, dv_upd = upd_layer(h, vec)
            h = norm_layer(h + ds_upd)
            vec = vec + dv_upd
            
        # 5. Readout
        return self.head(h, vec, role, mask, desc)
class FieldHead(nn.Module):
    def __init__(self, hidden, desc):
        super().__init__()
        self.desc_net=nn.Sequential(nn.Linear(desc,32),nn.SiLU(),nn.LayerNorm(32))
        self.net=nn.Sequential(nn.Linear(9*hidden+32,128),nn.SiLU(),nn.Dropout(.05),nn.Linear(128,64),nn.SiLU(),nn.Linear(64,1))
    def forward(self,h,v,role,mask,desc):
        copies,core=self.context
        def mean(value,m):
            return (value*m[...,None]).sum(1)/m.sum(1).clamp(min=1)[...,None]
        s=mask & (role==0); heavy=s & self.heavy
        hs=mean(h,s)
        solvent=torch.stack([mean(h,mask & (copies==k)) for k in (1,2,3)]).mean(0)
        vn=torch.linalg.vector_norm(v,dim=-1)
        vv=torch.stack([mean(vn,mask & (copies==k)) for k in (1,2,3)]).mean(0)
        size=(h*heavy[...,None]).sum(1)/heavy.sum(1).clamp(min=1).sqrt()[...,None]
        f=torch.cat([hs,size,mean(h,heavy & core),mean(h,heavy & ~core),solvent,hs*solvent,(hs-solvent).abs(),mean(vn,s),vv],-1)
        return self.net(torch.cat([f,self.desc_net(desc)],-1)).squeeze(-1)

class FieldModel(PaiNNModelV2):
    def __init__(self,field=False):
        super().__init__(96,3,19 if field else 16)
        self.field=field
        if field:self.head=FieldHead(96,19)
    def forward(self,z,pos,role,mask,desc,copies,core,edge_gate=None):
        if self.field:
            # Per-batch metadata is supplied immediately before the synchronous readout.
            self.head.context=(copies,core);self.head.heavy=z>1
        return super().forward(z,pos,role,mask,desc,edge_gate=edge_gate)

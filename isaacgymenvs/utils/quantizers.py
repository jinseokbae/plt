from typing import List, Optional
import torch
import torch.nn as nn
from vector_quantize_pytorch import VectorQuantize, ResidualVQ, GroupedResidualVQ, FSQ, FSNQ, LFQ, ResidualFSQ

def QuantizerSelector(quant_type, code_num, num_quants, dim):
    if quant_type == "simple":
        quantizer = SimpleQuantizer(
            dim=dim,
            codebook_size=code_num
        )
    elif quant_type == "basic":
        quantizer = VectorQuantize(
            dim=dim,
            codebook_size=code_num,
            decay=0.99,
            threshold_ema_dead_code=2,
            commitment_weight=1.0,
            kmeans_init=True,
        )
    elif quant_type == "basic_learnable":
        quantizer = VectorQuantize(
            dim=dim,
            codebook_size=code_num,
            ema_update=False,
            learnable_codebook=True,
            commitment_weight=1.0,
            kmeans_init=True,
        )
    elif quant_type == "basic_cosine":
        quantizer = VectorQuantize(
            dim=dim,
            codebook_size=code_num,
            decay=0.99,
            threshold_ema_dead_code=2,
            commitment_weight=1.0,
            kmeans_init=True,
            # cosine
            use_cosine_sim=True,
        )
    elif quant_type == "basic_ortho":
        quantizer = VectorQuantize(
            dim=dim,
            codebook_size=code_num,
            decay=0.99,
            threshold_ema_dead_code=2,
            commitment_weight=1.0,
            kmeans_init=True,
            # orthogonal
            orthogonal_reg_weight = 10,                 # in paper, they recommended a value of 10
            orthogonal_reg_max_codes = 128,             # this would randomly sample from the codebook for the orthogonal regularization loss, for limiting memory usage
            orthogonal_reg_active_codes_only = False    # set this to True if you have a very large codebook, and would only like to enforce the loss on the activated codes per batch
        )
    elif quant_type == "rvq":
        quantizer = ResidualVQ(
            dim=dim,
            codebook_size=code_num,
            decay=0.99,
            threshold_ema_dead_code=2,
            commitment_weight=1.0,
            kmeans_init=True,
            # rvq
            num_quantizers=num_quants,
            quantize_dropout=True,
        )
    elif quant_type == "rvq_no_dropout":
        quantizer = ResidualVQ(
            dim=dim,
            codebook_size=code_num,
            decay=0.99,
            threshold_ema_dead_code=2,
            commitment_weight=1.0,
            kmeans_init=True,
            # rvq
            num_quantizers=num_quants,
            quantize_dropout=False,
        )
    elif quant_type == "rvq_learnable":
        quantizer = ResidualVQ(
            dim=dim,
            codebook_size=code_num,
            ema_update=False,
            learnable_codebook=True,
            commitment_weight=1.0,
            kmeans_init=True,
            # rvq
            num_quantizers=num_quants,
            quantize_dropout=True,
        )
    elif quant_type == "rvq_ce":
        quantizer = ResidualVQ(
            dim=dim,
            codebook_size=code_num,
            decay=0.99,
            threshold_ema_dead_code=2,
            commitment_weight=1.0,
            kmeans_init=True,
            # rvq
            num_quantizers=num_quants,
            quantize_dropout=True,
            # ce
            commitment_use_cross_entropy_loss = True,
        )
    elif quant_type == "rvq_cosine":
        quantizer = ResidualVQ(
            dim=dim,
            codebook_size=code_num,
            decay=0.99,
            threshold_ema_dead_code=2,
            commitment_weight=1.0,
            kmeans_init=True,
            # rvq
            num_quantizers=num_quants,
            quantize_dropout=True,
            # cosine
            use_cosine_sim=True,
        )
    elif quant_type == "rqvq_cosine":
        quantizer = ResidualVQ(
            dim=dim,
            codebook_size=code_num,
            decay=0.99,
            threshold_ema_dead_code=2,
            commitment_weight=1.0,
            kmeans_init=True,
            # rvq
            num_quantizers=num_quants,
            quantize_dropout=True,
            # rqvq
            stochastic_sample_codes = True,
            sample_codebook_temp = 0.1,         # temperature for stochastically sampling codes, 0 would be equivalent to non-stochastic
            shared_codebook = True,              # whether to share the codebooks for all quantizers or not
            # cosine
            use_cosine_sim=True
        )
    elif quant_type == "group_rvq":
        quantizer = GroupedResidualVQ(
            dim=dim,
            codebook_size=code_num,
            decay=0.99,
            threshold_ema_dead_code=2,
            commitment_weight=1.0,
            kmeans_init=True,
            # rvq
            num_quantizers=num_quants,
            quantize_dropout=True,
            # grouped rvq
            groups=2,
        )
    elif quant_type in ["fsq", "fsnq", "rfsq"]:
        discrete_code_num = [2 ** 8, 2 ** 10, 2 ** 12, 2 ** 13, 2 ** 14, 2 ** 16]
        assert code_num in discrete_code_num
        levels = [
            [8, 6, 5],
            [8, 5, 5, 5],
            [7, 5, 5, 5, 5],
            [8, 8, 5, 5, 5],
            [8, 8, 8, 6, 5],
            [8, 8, 8, 5, 5, 5]
        ]
        idx = discrete_code_num.index(code_num)
        chosen_level = levels[idx]
        if quant_type == "fsq":
            quantizer = SimpleFSQ(
                dim=dim,
                levels=chosen_level # target codebook size : 1024, refer to original paper
            )
        elif quant_type == "rfsq":
            quantizer = SimpleResidualFSQ(
                dim=dim,
                levels=chosen_level, # target codebook size : 1024, refer to original paper
                num_quantizers=num_quants,
                quantize_dropout=True
            )
        else:
            quantizer = SimpleFSNQ(
                dim=dim,
                levels=chosen_level # target codebook size : 1024, refer to original paper
            )

    elif quant_type == "lfq":
        quantizer = LFQ(
            codebook_size=code_num,
            dim=dim,
        )
    return quantizer

class SimpleFSQOld(nn.Module):
    def __init__(
        self,
        levels: List[int],
        dim: int,
    ):
        super().__init__()
        self.fsq = FSQ(levels=levels, dim=dim)
    
    def forward(self, x):
        xhat, indices = self.fsq(x)
        null_loss = torch.zeros(1, device=x.device)
        return xhat, indices, null_loss
    
    def get_codes_from_indices(self, indices):
        xhat = self.fsq.indices_to_codes(indices)
        return xhat

class SimpleResidualFSQ(ResidualFSQ):
    def __init__(
        self,
        **kwargs
    ):
        super().__init__(**kwargs)
    
    def forward(self, x):
        xhat, indices = super().forward(x)
        null_loss = torch.zeros(1, device=x.device)
        return xhat, indices, null_loss
    
    def get_codes_from_indices(self, indices):
        xhat = self.indices_to_codes(indices)
        return xhat

class SimpleFSQ(FSQ):
    def __init__(
        self,
        levels: List[int],
        dim: Optional[int] = None,
        num_codebooks = 1,
        keep_num_codebooks_dim: Optional[bool] = None,
        scale: Optional[float] = None
    ):
        super().__init__(levels, dim, num_codebooks, keep_num_codebooks_dim, scale)
    
    def forward(self, x):
        xhat, indices = super().forward(x)
        null_loss = torch.zeros(1, device=x.device)
        return xhat, indices, null_loss
    
    def get_codes_from_indices(self, indices):
        xhat = self.indices_to_codes(indices)
        return xhat

class SimpleFSNQ(FSNQ):
    def __init__(
        self,
        levels: List[int],
        dim: Optional[int] = None,
        num_codebooks = 1,
        keep_num_codebooks_dim: Optional[bool] = None,
        scale: Optional[float] = None
    ):
        super().__init__(levels, dim, num_codebooks, keep_num_codebooks_dim, scale)
    
    def forward(self, x):
        xhat, indices = super().forward(x)
        null_loss = torch.zeros(1, device=x.device)
        return xhat, indices, null_loss

class SimpleQuantizer(nn.Module):
    def __init__(self, dim, codebook_size):
        super().__init__()
        self.embeddings = nn.Embedding(codebook_size, dim)
        self.embeddings.weight.data.uniform_(-1 / codebook_size,  1 / codebook_size)
    
    def forward(self, z):
        flat_z = z.view(-1, z.shape[-1])
        encoding_indices = self.get_code_indices(flat_z)
        quantized = self.quantize(encoding_indices)
        quantized = quantized.view(*z.shape)
        e_latent_loss = torch.mean((quantized.detach() - z).pow(2).sum(dim=-1))
        q_latent_loss = torch.mean((quantized - z.detach()).pow(2).sum(dim=-1))
        commit_loss = e_latent_loss + 0.25 * q_latent_loss
        z_curr = z + (quantized - z).detach()
        encoding_indices = encoding_indices[:, None]
        return z_curr, encoding_indices, commit_loss
    
    def get_code_indices(self, flat_x):
        distances = (
                torch.sum(flat_x ** 2, dim=1, keepdim=True) +
                torch.sum(self.embeddings.weight ** 2, dim=1) -
                2. * torch.matmul(flat_x, self.embeddings.weight.t())
        )
        encoding_indices = torch.argmin(distances, dim=1)
        return encoding_indices
    
    def get_codes_from_indices(self, indices):
        return self.quantize(indices)

    def quantize(self, encoding_indices):
        return self.embeddings(encoding_indices)

    @property
    def codebooks(self,):
        return self.embeddings.weight.data
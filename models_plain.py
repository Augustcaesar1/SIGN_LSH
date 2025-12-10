import torch
import torch.nn.functional as F
from lsh_core import LSHBase, fast_walsh_hadamard_transform


# --- 模型 A: 单表随机矩阵 ---
class RandomMatrixRetriever(LSHBase):
    def __init__(self, input_dim, total_bits, device='cpu'):
        super().__init__(device)
        self.total_bits = total_bits
        self.R = torch.randn(input_dim, total_bits, device=device)

    def compute_fingerprint(self, x):
        proj = torch.mm(x, self.R)
        sign_bits = (proj >= 0)
        fingerprints = []
        for i in range(0, self.total_bits, 64):
            chunk = sign_bits[:, i:i + 64]
            if chunk.shape[1] < 64: chunk = F.pad(chunk, (0, 64 - chunk.shape[1]))
            fingerprints.append(self._pack_bits(chunk).unsqueeze(1))
        return torch.cat(fingerprints, dim=1)

    def train(self, db):
        self.db_fingerprints = self.compute_fingerprint(db)

    def query(self, qs, k):
        q = self.compute_fingerprint(qs)
        return self.query_with_fingerprints(q, k)


# --- 模型 B: 多表随机矩阵 ---
class MultiTableRandomRetriever(LSHBase):
    def __init__(self, input_dim, total_bits, num_tables=4, device='cpu'):
        super().__init__(device)
        self.num_tables = num_tables
        self.bits_per_table = total_bits // num_tables
        self.Rs = torch.randn(num_tables, input_dim, self.bits_per_table, device=device)

    def compute_fingerprint(self, x):
        proj = torch.einsum('bd, tdk -> btk', x, self.Rs)
        sign_bits = (proj >= 0)
        fingerprints = []
        for i in range(0, self.bits_per_table, 64):
            chunk = sign_bits[:, :, i:i + 64]
            if chunk.shape[2] < 64: chunk = F.pad(chunk, (0, 0, 0, 64 - chunk.shape[2]))
            fingerprints.append(self._pack_bits(chunk).unsqueeze(-1))
        return torch.cat(fingerprints, dim=-1)

    def train(self, db):
        self.db_fingerprints = self.compute_fingerprint(db)

    def query(self, qs, k):
        q = self.compute_fingerprint(qs)
        return self.query_with_fingerprints(q, k)


# --- 模型 C: 基础 FWHT ---
class FastHadamardRetriever(LSHBase):
    def __init__(self, input_dim, total_bits, device='cpu'):
        super().__init__(device)
        self.total_bits = total_bits
        self.h_dim = 1
        while self.h_dim < max(input_dim, total_bits): self.h_dim *= 2
        self.D = (torch.randint(0, 2, (self.h_dim,), device=device) * 2 - 1).float()
        self.perm = torch.randperm(self.h_dim, device=device)[:total_bits]

    def compute_fingerprint(self, x):
        B, D_in = x.shape
        x_pad = F.pad(x, (0, self.h_dim - D_in)) if D_in < self.h_dim else x[:, :self.h_dim]
        x_rot = x_pad * self.D
        proj = fast_walsh_hadamard_transform(x_rot, normalize=False)
        proj = proj[:, self.perm]
        sign_bits = (proj >= 0)
        fingerprints = []
        for i in range(0, self.total_bits, 64):
            chunk = sign_bits[:, i:i + 64]
            if chunk.shape[1] < 64: chunk = F.pad(chunk, (0, 64 - chunk.shape[1]))
            fingerprints.append(self._pack_bits(chunk).unsqueeze(1))
        return torch.cat(fingerprints, dim=1)

    def train(self, db):
        self.db_fingerprints = self.compute_fingerprint(db)

    def query(self, qs, k):
        q = self.compute_fingerprint(qs)
        return self.query_with_fingerprints(q, k)


# --- 模型 D: 优化多表 FWHT (SecretOptimizedFastHadamardRetriever 的基础) ---
class OptimizedFastHadamardRetriever(LSHBase):
    def __init__(self, input_dim, total_bits, num_tables=4, device='cpu'):
        super().__init__(device)
        self.num_tables = num_tables
        self.bits_per_table = total_bits // num_tables
        self.h_dim = 1
        while self.h_dim < input_dim: self.h_dim *= 2

        # 参数初始化
        self.D_diags = (torch.randint(0, 2, (num_tables, self.h_dim), device=device) * 2 - 1).float()
        perms = []
        for _ in range(num_tables):
            perms.append(torch.randperm(self.h_dim, device=device)[:self.bits_per_table])
        self.perms = torch.stack(perms)

    def compute_fingerprint(self, x):
        B, D_in = x.shape
        x_pad = F.pad(x, (0, self.h_dim - D_in)) if D_in < self.h_dim else x[:, :self.h_dim]

        # 扩展并旋转: (B, 1, H) * (T, H) -> (B, T, H)
        x_pad = x_pad.unsqueeze(1)
        x_rot = x_pad * self.D_diags.unsqueeze(0)

        # FWHT
        x_flat = x_rot.view(-1, self.h_dim)
        proj_flat = fast_walsh_hadamard_transform(x_flat, normalize=False)
        proj = proj_flat.view(B, self.num_tables, self.h_dim)

        # 采样 Permutation
        gather_idx = self.perms.unsqueeze(0).expand(B, -1, -1)
        final_proj = torch.gather(proj, 2, gather_idx)

        sign_bits = (final_proj >= 0)
        fingerprints = []
        for i in range(0, self.bits_per_table, 64):
            chunk = sign_bits[:, :, i:i + 64]
            if chunk.shape[2] < 64: chunk = F.pad(chunk, (0, 0, 0, 64 - chunk.shape[2]))
            fingerprints.append(self._pack_bits(chunk).unsqueeze(-1))
        return torch.cat(fingerprints, dim=-1)

    def train(self, db):
        self.db_fingerprints = self.compute_fingerprint(db)

    def query(self, qs, k):
        q = self.compute_fingerprint(qs)
        return self.query_with_fingerprints(q, k)
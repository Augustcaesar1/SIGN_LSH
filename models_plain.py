import torch
import torch.nn.functional as F
from lsh_core import fast_walsh_hadamard_transform

class LSHBase:
    def __init__(self, device='cpu'):
        self.device = device
        self.db_fingerprints = None

    def _pack_bits(self, binary_tensor):
        x = binary_tensor.long()
        packed = torch.zeros(x.shape[:-1], dtype=torch.int64, device=self.device)
        val = torch.tensor(1, dtype=torch.int64, device=self.device)
        for i in range(x.shape[-1]):
            packed += x[..., i] * val
            val <<= 1
        return packed

    def _popcount(self, t):
        """兼容 PyTorch 旧版本的汉明重量计算 (SWAR算法)"""
        if hasattr(t, "bit_count"): return t.bit_count()
        
        x = t
        mask1 = torch.tensor(0x5555555555555555, dtype=torch.int64, device=t.device)
        x = (x & mask1) + ((x >> 1) & mask1)
        mask2 = torch.tensor(0x3333333333333333, dtype=torch.int64, device=t.device)
        x = (x & mask2) + ((x >> 2) & mask2)
        mask4 = torch.tensor(0x0f0f0f0f0f0f0f0f, dtype=torch.int64, device=t.device)
        x = (x & mask4) + ((x >> 4) & mask4)
        mask8 = torch.tensor(0x00ff00ff00ff00ff, dtype=torch.int64, device=t.device)
        x = (x & mask8) + ((x >> 8) & mask8)
        mask16 = torch.tensor(0x0000ffff0000ffff, dtype=torch.int64, device=t.device)
        x = (x & mask16) + ((x >> 16) & mask16)
        mask32 = torch.tensor(0x00000000ffffffff, dtype=torch.int64, device=t.device)
        x = (x & mask32) + ((x >> 32) & mask32)
        return x

    def query_with_fingerprints(self, query_fp, k=100):
        if self.db_fingerprints is None: raise ValueError("Run train(db) first!")
        
        q = query_fp.unsqueeze(1)
        db = self.db_fingerprints.unsqueeze(0)
        
        xor_result = torch.bitwise_xor(q, db)
        hamming_dist = self._popcount(xor_result).sum(dim=(-1, -2))
        
        return torch.topk(hamming_dist, k, largest=False)

# ==========================================
# 3. 统一检索器 (支持 Hadamard 和 Random 投影)
# ==========================================
class UnifiedLSHRetriever(LSHBase):
    def __init__(self, input_dim, total_bits, num_tables=1, projection_type='hadamard', use_fast_transform=True, device='cpu'):
        super().__init__(device)
        self.input_dim = input_dim
        self.num_tables = num_tables
        self.bits_per_table = total_bits // num_tables
        self.projection_type = projection_type.lower()
        self.use_fast_transform = use_fast_transform
        
        # --- 分支 1: Hadamard 投影 (结构化) ---
        if self.projection_type == 'hadamard':
            # Padding 维度计算
            self.proj_dim = 1
            while self.proj_dim < input_dim: self.proj_dim *= 2
            
            # 随机翻转矩阵 D
            self.D = (torch.randint(0, 2, (num_tables, self.proj_dim), device=device) * 2 - 1).float()
            
            # 采样置换 Perms
            perms = [torch.randperm(self.proj_dim, device=device)[:self.bits_per_table] for _ in range(num_tables)]
            self.perms = torch.stack(perms)
            
            # 如果不使用算子，显式构建矩阵
            if not self.use_fast_transform:
                print(f"[Info] Building explicit Hadamard matrix ({self.proj_dim}x{self.proj_dim})...")
                self.H = self._build_explicit_hadamard(self.proj_dim, device)

        # --- 分支 2: Random 投影 (非结构化) ---
        elif self.projection_type == 'random':
            self.proj_dim = input_dim 
            self.R = torch.randn(num_tables, input_dim, self.bits_per_table, device=device)
        else:
            raise ValueError(f"Unknown projection_type: {projection_type}")

    def _build_explicit_hadamard(self, n, device):
        """递归构建显式矩阵"""
        if n == 1: return torch.tensor([[1.]], device=device)
        h = self._build_explicit_hadamard(n // 2, device)
        return torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)

    def _project(self, x):
        B = x.shape[0]
        
        if self.projection_type == 'hadamard':
            # 1. Padding
            if x.shape[1] < self.proj_dim: 
                x = F.pad(x, (0, self.proj_dim - x.shape[1]))
            
            # 2. 随机翻转 (Rotate)
            x_rot = x.unsqueeze(1) * self.D.unsqueeze(0) # (B, T, Dim)

            # 3. 变换 (Transform)
            if self.use_fast_transform:
                # 算子加速 O(N log N)
                proj_full = fast_walsh_hadamard_transform(x_rot.view(-1, self.proj_dim), False)
                proj_full = proj_full.view(B, self.num_tables, self.proj_dim)
            else:
                # 矩阵乘法 O(N^2)
                proj_full = torch.matmul(x_rot, self.H)
            
            # 4. 采样
            gather_idx = self.perms.unsqueeze(0).expand(B, -1, -1)
            return torch.gather(proj_full, 2, gather_idx)

        elif self.projection_type == 'random':
            # 矩阵乘法
            return torch.einsum('bd, tdk -> btk', x, self.R)

    def compute_fingerprint(self, x):
        proj = self._project(x)
        sign_bits = (proj >= 0)
        fingerprints = []
        for i in range(0, self.bits_per_table, 64):
            chunk = sign_bits[:, :, i:i + 64]
            if chunk.shape[2] < 64: chunk = F.pad(chunk, (0, 64 - chunk.shape[2]))
            fingerprints.append(self._pack_bits(chunk).unsqueeze(-1))
        return torch.cat(fingerprints, dim=-1)

    def train(self, db): self.db_fingerprints = self.compute_fingerprint(db)
    def query(self, qs, k=100): return self.query_with_fingerprints(self.compute_fingerprint(qs), k)

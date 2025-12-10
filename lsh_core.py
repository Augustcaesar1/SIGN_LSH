import torch
import torch.nn.functional as F
import jax.numpy as jnp


# --- 核心数学函数 ---

def fast_walsh_hadamard_transform(x, normalize=True):
    """ PyTorch 版 FWHT """
    B, N = x.shape
    if (N & (N - 1)) != 0:
        raise ValueError(f"Dim {N} must be power of 2 for FWHT.")
    out = x.clone()
    h = 1
    while h < N:
        out = out.view(B, N // (2 * h), 2, h)
        x1 = out[:, :, 0, :]
        x2 = out[:, :, 1, :]
        out = torch.cat((x1 + x2, x1 - x2), dim=2)
        h *= 2
    out = out.view(B, N)
    if normalize:
        out = out / torch.sqrt(torch.tensor(N, float))
    return out


def sf_fwht(x, normalize=False):
    """ JAX/SPU 版 FWHT """
    shape = x.shape
    N = shape[-1]
    x_flat = x.reshape(-1, N)
    B_total = x_flat.shape[0]
    out = x_flat
    h = 1
    while h < N:
        out = out.reshape(B_total, N // (2 * h), 2, h)
        x1 = out[:, :, 0, :]
        x2 = out[:, :, 1, :]
        out = jnp.concatenate((x1 + x2, x1 - x2), axis=2)
        h *= 2
    out = out.reshape(shape)
    if normalize:
        out = out / jnp.sqrt(N)
    return out


# --- LSH 基类 ---

class LSHBase:
    def __init__(self, device):
        self.device = device
        self.bit_powers = (2 ** torch.arange(64, device=device)).to(torch.int64)
        self.db_fingerprints = None

    def _pack_bits(self, bool_tensor):
        """ 将 bool tensor 压缩为 int64 """
        return (bool_tensor.to(torch.int64) * self.bit_powers).sum(dim=-1)

    def _popcount_swar(self, n):
        """ SWAR 算法计算汉明重量 """
        n = n - ((n >> 1) & 0x5555555555555555)
        n = (n & 0x3333333333333333) + ((n >> 2) & 0x3333333333333333)
        n = (n + (n >> 4)) & 0x0f0f0f0f0f0f0f0f
        n = (n * 0x0101010101010101) >> 56
        return n

    def query_with_fingerprints(self, q_fingerprints, k):
        """ 使用预计算的指纹进行查询 """
        is_multi_table = (q_fingerprints.dim() == 3)
        # 广播异或
        xor = torch.bitwise_xor(q_fingerprints.unsqueeze(1), self.db_fingerprints.unsqueeze(0))

        if is_multi_table:
            # 多表求和: (Batch, DB_Size, Tables) -> (Batch, DB_Size)
            dists = self._popcount_swar(xor).sum(dim=-1).sum(dim=-1)
        else:
            # 单表求和: (Batch, DB_Size, Chunks) -> (Batch, DB_Size)
            dists = self._popcount_swar(xor).sum(dim=2)

        return torch.topk(dists, k=k, dim=1, largest=False)